# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Implements kernel functions for volumetric primitives in Mitsuba scenes,
including Gaussian and Epanechnikov kernels for density evaluation. Also provides
an implementation of the primitive tracing algorithm described in the paper.
'''

from __future__ import annotations # Delayed parsing of type annotations
from dataclasses import dataclass

import drjit as dr
import mitsuba as mi

from .stack import alloc_stack

#-------------------------------------------------------------------------------

def get_ellipsoids_shape(scene, requested=True):
    for i, shape in enumerate(scene.shapes()):
        if shape.shape_type() == +mi.ShapeType.Ellipsoids:
            if i > 0:
                mi.Log(mi.LogLevel.Warn, 'For performance consideration, it is '
                                         'better to define the ellipsoids shape '
                                         'as the first shape in the scene!')
            return shape, mi.ShapePtr(scene.shapes_dr()[i])
    if requested:
        raise Exception("Couldn't find ellipsoids shape in the scene!")
    else:
        return None, mi.ShapePtr(None)

def check_ellipsoids_attribute(scene, attributes):
    for i, shape in enumerate(scene.shapes()):
        if shape.shape_type() == +mi.ShapeType.Ellipsoids:
            for attr in attributes:
                assert shape.has_attribute(attr), f'Requested ellipsoid attribute \'{attr}\' not found on shape \'{shape.id()}\''

def has_ellipsoids_shapes(scene):
    for i, shape in enumerate(scene.shapes()):
        if shape.shape_type() == +mi.ShapeType.Ellipsoids:
            return True
    return False

@dataclass
class Ellipsoid:
    center:  mi.Point3f = mi.Point3f(0)
    scale:   mi.Vector3f = mi.Vector3f(0)
    quat:    mi.Quaternion4f = mi.Quaternion4f(0)
    rot:     mi.Matrix3f = mi.Matrix3f(0)
    extent:  mi.Float = mi.Float(3.0)

    @staticmethod
    def ravel(center, scale, quat):
        data = dr.empty(mi.Float, dr.width(center) * 10)
        idx = dr.arange(mi.UInt32, dr.width(center))
        for i in range(3):
            dr.scatter(data, center[i], idx * 10 + i)
        for i in range(3):
            dr.scatter(data, scale[i], idx * 10 + 3 + i)
        for i in range(4):
            dr.scatter(data, quat[i], idx * 10 + 6 + i)
        return data

    @staticmethod
    def unravel(data):
        idx = dr.arange(mi.UInt32, dr.width(data) // 10)
        center = mi.Point3f([dr.gather(mi.Float, data, idx * 10 + 0 + i) for i in range(3)])
        scale  = mi.Vector3f([dr.gather(mi.Float, data, idx * 10 + 3 + i) for i in range(3)])
        quat   = mi.Quaternion4f([dr.gather(mi.Float, data, idx * 10 + 6 + i) for i in range(4)])
        rot    = dr.quat_to_matrix(quat, size=3)
        return Ellipsoid(center, scale, quat, rot, extent=None)

    @staticmethod
    def gather(shape, prim_index, active):
        def func(self, prim_index, active):
            if self is not None and self.shape_type() == +mi.ShapeType.Ellipsoids:
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_index
                data = self.eval_attribute_x("ellipsoid", si, active)
                center = mi.Point3f([data[i] for i in range(3)])
                scale  = mi.Vector3f([data[i + 3] for i in range(3)])
                quat   = mi.Quaternion4f([data[i + 6] for i in range(4)])
                rot    = dr.quat_to_matrix(quat, size=3)
                extent = self.eval_attribute_1("extent", si, active)
                return center, scale, quat, rot, extent
            else:
                return mi.Point3f(0), mi.Vector3f(0), mi.Quaternion4f(0), mi.Matrix3f(0), mi.Float(0)
        return Ellipsoid(*dr.dispatch(shape, func, prim_index, active))

#-------------------------------------------------------------------------------

class Kernel:
    @staticmethod
    def factory(props: mi.Properties):
        name = props.get('kernel_type', 'gaussian')

        if name == 'gaussian':
            return GaussianKernel(props)
        elif name == 'epanechnikov':
            return EpanechnikovKernel(props)
        else:
            raise Exception('Unknown kernel type! Should be one of "gaussian", "triangle" or "epanechnikov".')

    def __init__(self, props):
        self.type = props.get('kernel_type', 'gaussian')

        # Whether to normalize the density integral (i.e., kernel_normalized=True means that peak value is 1.0)
        self.normalized = props.get('kernel_normalized', False)

        # Whether we should always integrate the full density kernel (used in `gaussian_sp` integrator)
        self.full_range = props.get('kernel_full_range', False)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def normalization_factor(self, ellipsoid: Ellipsoid) -> mi.Float:
        raise Exception('Not implemented!')

#-------------------------------------------------------------------------------

class GaussianKernel(Kernel):
    def __init__(self, props):
        super().__init__(props)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s
        return dr.exp(-0.5 * dr.sum(p * p))

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s

        denom = dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(s)
        pdf = dr.exp(-0.5 * dr.sum(p * p)) / denom
        return dr.select(active, pdf, 0.0)

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        s = ellipsoid.scale
        w = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        w_norm = dr.norm(w)
        w /= w_norm

        B = dr.dot(w, p)
        C = dr.dot(p, p)

        K = sigmat * dr.rcp(w_norm) * dr.exp(-0.5 * (C - B * B)) * dr.rcp(2.0 * dr.pi) * dr.sqrt(dr.prod(s))
        erfinv_arg = dr.erf(B / dr.sqrt(2.0)) + 2.0 * chi / K
        t_s = dr.sqrt(2.0) * dr.erfinv(erfinv_arg) - B
        t_s = dr.select(dr.isnan(t_s), dr.inf, t_s)  # Handle NaNs arising from intersection points in the infinite (dr.abs(erfinv_arg)>1)
        # TODO: would be interesting that erfinv returns -inf or inf instead of nans, could we change this behavior?
        
        return dr.select(active, t_s, 0.0)

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        if self.full_range or (tmin is None and tmax is None):
            w = ellipsoid.rot.T * ray.d
            p = ellipsoid.rot.T * (ray.o - ellipsoid.center)
            s = ellipsoid.scale
            w /= s
            w_norm = dr.norm(w)
            w /= w_norm
            p /= s

            B = dr.dot(w, p)
            C = dr.dot(p, p)
            density = dr.sqrt(2.0 * dr.pi) * dr.rcp(w_norm) * dr.exp(-0.5 * (C - B * B))
        else:
            active = mi.Bool(active) & (tmin < tmax) & (tmax > 0.0) # Catching rare edge-cases and avoiding (NaNs)
            s = ellipsoid.scale
            w = ellipsoid.rot.T * ray.d / s
            p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
            w_norm = dr.norm(w)
            w /= w_norm
            t_sim = 0.5 * (tmax - tmin)
            t_sim *= w_norm
            p = p + t_sim * w

            B = dr.dot(w, p)
            C = dr.dot(p, p)
            erf = dr.erf(t_sim * dr.sqrt(0.5))
            density = dr.rcp(w_norm) * dr.exp(-0.5 * (C - B * B)) * erf * dr.sqrt(2.0 * dr.pi)
        
        if self.normalized:
            density *= self.normalization_factor(s)

        density = dr.maximum(density, 0.0)
        density[~active] = 0.0
        density[~dr.isfinite(density)] = 0.0

        return density                                      

    def normalization_factor(self, s: mi.Vector3f) -> mi.Float:
        return dr.rcp(dr.sqrt(8.0 * (dr.pi * dr.pi * dr.pi)) * dr.prod(s))
#-------------------------------------------------------------------------------

class EpanechnikovKernel(Kernel):
    def __init__(self, props):
        super().__init__(props)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        
        s = ellipsoid.scale
        p = ellipsoid.rot.T * (p - ellipsoid.center) / s

        dist = dr.norm(p)
        value = (1.0 - (1.0/7.0) * dist**2)
        return dr.maximum(value, 0.0)

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        
        s = ellipsoid.scale
        p = ellipsoid.rot.T * (p - ellipsoid.center) / s

        dist = dr.norm(p)
        K_norm = self.normalization_factor(ellipsoid)
        density = K_norm * (1.0 - (1.0/7.0) * (dist ** 2)) 

        return dr.select(active, dr.clip(density, 0.0, 1.0), 0.0)

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        raise Exception('EpanechnikovKernel.inv_cdf(): Not implemented!')

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        w = ellipsoid.rot.T * ray.d
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center)
        s = ellipsoid.scale
        w /= s
        w_norm = dr.norm(w)
        w /= w_norm
        p /= s
        
        if self.full_range or (tmin is None and tmax is None):
            valid, tmin, tmax = ray_ellipsoid_intersection(ray, ellipsoid, active)
            active = mi.Bool(active) & valid

        active = mi.Bool(active) & (tmin < tmax) & (tmax > 0.0) # Catching rare edge cases and avoiding (NaNs)

        C1 = 3.0 * (((p.x ** 2 - 7.0) + p.y ** 2) + p.z ** 2)
        C2 = 3.0 * dr.sum(p * w)
        density = dr.rcp(w_norm) * ((tmin**3 + C2 * tmin**2 + C1 * tmin) - (tmax**3 + C2 * tmax**2 + C1 * tmax))

        if self.normalized:
            density *= self.normalization_factor(ellipsoid)

        density[~active] = 0.0
        density = dr.maximum(density, 0.0)
        density[~dr.isfinite(density)] = 0.0

        return density
    
    def normalization_factor(self, ellipsoid: Ellipsoid) -> mi.Float:
        # Analytical formula for isotropic Epanechnikov maximum density integral value
        s = ellipsoid.scale
        return 15/(84 * dr.pi * dr.prod(s))
#-------------------------------------------------------------------------------

def safe_rcp(x):
    return dr.select((x != 0.0), dr.rcp(x), 0.0)

def spawn_ray_origin(si, ray, epsilon = 5e-6):
    '''
    Spawn ray from volumetric primitive and return new ray origin
    '''
    return si.p + ray.d * epsilon

def ray_ellipsoid_intersection(ray: mi.Ray3f,
                               ellipsoid: Ellipsoid,
                               active: mi.Bool):
    scale = ellipsoid.scale * ellipsoid.extent

    d = ellipsoid.rot.T * ray.d / scale
    o = ellipsoid.rot.T * (ray.o - ellipsoid.center) / scale

    if False:
        a = dr.squared_norm(d)
        b = 2.0 * dr.dot(o, d)
        c = dr.squared_norm(o) - 1.0
        valid, near_t, far_t = mi.math.solve_quadratic(a, b, c)
    else:
        # Taken from Ray Tracing Gems 2
        a = dr.squared_norm(d)
        b = -dr.dot(o, d)
        c = dr.squared_norm(o) - 1.0
        discr = 1.0 - dr.squared_norm(o + (b / a) * d)
        valid, near_t, far_t = mi.math.improved_solve_quadratic(a, b, c, discr)

    return valid & active, near_t, far_t

@dataclass
class PrimitiveID:
    shape: mi.ShapePtr = mi.ShapePtr(None)
    index: mi.UInt32 = mi.UInt32(0)

@dr.syntax
def primitive_tracing(scene :mi.Scene,
                      sampler: mi.Sampler,
                      ray: mi.Ray3f,
                      primitives: dr.Local[PrimitiveID],
                      depth: mi.UInt32,
                      callback,
                      payload,
                      active: mi.Bool,
                      max_depth_primitive: int=-1,
                      rr_depth_primitive: int=-1):
    '''
    Implements the Primitive Tracing algorithm that iterates over segments
    defined by kernel intersections along a ray, accounting for overlaps of
    primitives. A specified callback is called to process each segment. This
    routine is used in the `volprim_prb` integrator to implement volume
    interaction sampling.
    '''
    @dataclass
    class PrimitiveVertex:
        shape: mi.ShapePtr = mi.ShapePtr(None)
        index: mi.UInt32 = mi.UInt32(0)
        t: mi.Float = mi.Float(dr.inf)

    active    = mi.Bool(active)
    ray       = mi.Ray3f(ray)
    accum_t   = mi.Float(0.0)
    weight_rr = mi.Float(1.0)

    # Early exit if the scene doesn't contains volumetric primitives
    if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
        si = scene.ray_intersect(
            ray=ray,
            coherent=(depth == 0),
            ray_flags=(mi.RayFlags.All | mi.RayFlags.BackfaceCulling),
            active=active
        )
        return primitives, si, mi.Float(1.0), *payload

    # Stack of vertices to process as segments (allocate one more than max number of primitives)
    vertices = alloc_stack(PrimitiveVertex, mi.UInt32, primitives.alloc_size()+1)

    # Count primitives we have interacted with in the algorithm so far
    primitive_depth = mi.UInt32(0)

    # Surface interaction when intersecting other shapes
    surface_si = dr.zeros(mi.SurfaceInteraction3f)

    active1 = mi.Bool(active)
    while active1:
        # ----------------------------------------------------------------------
        # Create stack of vertices to be processed and remove invalid primitives
        # ----------------------------------------------------------------------

        vertices.clear(active1)
        active2 = mi.Bool(active1) & ~primitives.is_empty()
        count = mi.UInt32(0)
        it2   = mi.UInt32(0)
        while active2:
            prim_id = primitives.value(it2, active2)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active2)
            valid, t0, t1 = ray_ellipsoid_intersection(ray, ellipsoid, active2) # TODO should primitives be processed every time?
            primitives.write(prim_id, count, valid & (count != it2))
            vertices.push(PrimitiveVertex(prim_id.shape, prim_id.index, t1), valid)
            count[valid] += 1
            it2 += 1
            active2 &= (it2 < primitives.size())
        primitives.resize(count, active1)

        # ----------------------------------------------------------------------
        # Find next intersection
        # ----------------------------------------------------------------------

        # Perform Russian Roulette
        if rr_depth_primitive > 0:
            q = dr.minimum(weight_rr, 0.9)
            perform_rr = (primitive_depth > rr_depth_primitive)
            valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
            weight_rr[active1 & perform_rr] *= dr.rcp(q)
            weight_rr[active1 & ~valid_rr] = 0.0
            active1 &= valid_rr

        # Kill paths that are too long
        if max_depth_primitive > 0:
            weight_rr[active1 & (primitive_depth >= max_depth_primitive)] = 0.0
            active1 &= (primitive_depth < max_depth_primitive)

        si = scene.ray_intersect(
            ray=ray,
            coherent=(primitive_depth == 0),
            ray_flags=(mi.RayFlags.All | mi.RayFlags.BackfaceCulling),
            active=active1
        )

        is_ellipsoids = (si.shape.shape_type() == +mi.ShapeType.Ellipsoids)
        hit_surface   = active & si.is_valid() & ~is_ellipsoids
        hit_primitive = active & si.is_valid() &  is_ellipsoids

        surface_si[active1] = si
        surface_t = dr.select(hit_surface, si.t, dr.inf)

        vertices.push(PrimitiveVertex(si.shape, si.prim_index, si.t), hit_primitive)


        # ----------------------------------------------------------------------
        # Process segments up to this intersection
        # ----------------------------------------------------------------------

        seg_t0 = mi.Float(0.0)
        it3 = mi.UInt32(0)
        active3 = mi.Bool(active1) & ~primitives.is_empty()
        while active3:
            # ------------------------------------------------------------------
            # Find the closest vertex greater than the segment t0
            # ------------------------------------------------------------------

            it4 = mi.UInt32(0)
            active4 = mi.Bool(active3)
            vertex = PrimitiveVertex(mi.ShapePtr(None), mi.UInt32(0), mi.Float(dr.inf))
            while active4:
                v = vertices.value(it4, active4)
                is_closer = active4 & (v.t > seg_t0) & (v.t < vertex.t)
                vertex.shape[is_closer] = v.shape
                vertex.index[is_closer] = v.index
                vertex.t[is_closer] = v.t
                it4 += 1
                active4 &= (it4 < vertices.size())

            # This next vertex is the segment farther bound
            reached_surface = vertex.t > surface_t
            seg_t1 = dr.select(reached_surface, surface_t, vertex.t)

            # ------------------------------------------------------------------
            # Call segment callback
            # ------------------------------------------------------------------

            active_c, payload = callback(payload, primitives, ray, accum_t, seg_t0, seg_t1, active3)
            # TODO remove accum_t and set seg_t0/seg_t1 relative to original ray

            # Check whether the callback requested stopping here
            active3 &= active_c
            active1 &= active_c

            # ------------------------------------------------------------------
            # Remove vertex.index from primitives if necessary
            # ------------------------------------------------------------------

            # Check whether we have reach the primitive/surface we intersected last
            reached_si = si.is_valid() & (((si.prim_index == vertex.index) & (si.shape == vertex.shape)) | reached_surface)

            should_remove = mi.Bool(active3) & ~reached_si
            active6 = mi.Bool(should_remove) & (primitives.size() > 1)
            shift = mi.Bool(False)
            it6 = mi.UInt32(0)
            while active6:
                p = primitives.value(it6, active6)
                shift |= active6 & (p.shape == vertex.shape) & (p.index == vertex.index)
                primitives.write(primitives.value(it6 + 1, shift), it6, shift)
                it6 += 1
                active6 &= (it6 < (primitives.size() - 1))
            primitives.pop(should_remove)

            # ------------------------------------------------------------------
            # Continue with next segment
            # ------------------------------------------------------------------

            # Break if we are entering the primitive that we intersected last
            active3 &= ~reached_si

            # Otherwise continue with the next segment
            seg_t0[active3] = mi.Float(seg_t1)

            # Check whether we have already visited all the vertices
            it3 += 1
            active3 &= (it3 < vertices.size())

        # Stop if we processed all vertices and ray escaped or hit a surface
        active1 &= hit_primitive

        # Add primitive we are entering into to `primitives`
        valid_alloc = active1 & (primitives.size() < (primitives.alloc_size()))

        primitives.push(PrimitiveID(si.shape, si.prim_index), valid_alloc)
        primitive_depth[active1] += 1

        ray.o = spawn_ray_origin(si, ray)
        accum_t += si.t

    return primitives, surface_si, weight_rr, *payload

#-------------------------------------------------------------------------------

class EllipsoidsFactory:
    '''
    Helper class to build Ellipsoids datasets for testing purposes
    '''
    def __init__(self):
        self.centers = []
        self.scales = []
        self.quaternions = []
        self.sigmats = []
        self.albedos = []

    def add(self, mean, scale, sigmat=1.0, albedo=1.0, euler=[0.0, 0.0, 0.0]):
        self.centers.append(mi.ScalarPoint3f(mean))
        self.scales.append(mi.ScalarVector3f(scale))
        quaternion = dr.slice(dr.euler_to_quat(dr.deg2rad(mi.ScalarPoint3f(euler))))
        self.quaternions.append(mi.ScalarQuaternion4f(quaternion))
        self.sigmats.append(sigmat)
        if isinstance(albedo, float):
            albedo = mi.ScalarColor3f(albedo)
        self.albedos.append(albedo)

    def build(self):
        import numpy as np
        num_gaussians = len(self.centers)
        centers = mi.TensorXf(np.ravel(np.array(self.centers)), shape=(num_gaussians, 3))
        scales  = mi.TensorXf(np.ravel(np.array(self.scales)), shape=(num_gaussians, 3))
        quats   = mi.TensorXf(np.ravel(np.array(self.quaternions)), shape=(num_gaussians, 4))
        sigmats = mi.TensorXf(self.sigmats, shape=(num_gaussians, 1))
        self.albedos = np.array(self.albedos).reshape((num_gaussians, -1))
        albedos = mi.TensorXf(np.array(self.albedos))
        return centers, scales, quats, sigmats, albedos
