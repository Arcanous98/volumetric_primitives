# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator
from .common import *

class VolumetricPrimitiveRadianceFieldIntegrator(RBIntegrator):
    '''
    This plugin implements a simple radiance field integrator for ellipsoids shapes.

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        rr_depth (int): Minimum path depth before enabling the Russian roulette path termination.
        kernel_type (str): Name of the kernel to use for rendering the volumetric primitives, one of ['gaussian', 'epanechnikov'].
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", 64))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces

        rr_depth = int(props.get('rr_depth', -1))
        if rr_depth < 0 and rr_depth != -1:
            raise Exception("\"rr_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.rr_depth = mi.UInt32(rr_depth if rr_depth != -1 else 0xffffffff)

        # Check whether Russian Roulette should be enabled
        self.use_rr = rr_depth >= 0 and (rr_depth < max_depth or max_depth == -1)

        self.srgb_primitives = props.get('srgb_primitives', True)

        # Those kernel parameters are required for the volumetric transmittance model
        props['kernel_full_range'] = True
        props['kernel_normalized'] = True
        self.kernel = Kernel.factory(props)

    def traverse(self, callback):
        callback.put_parameter("max_depth",       self.max_depth,       mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("rr_depth",        self.rr_depth,        mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('srgb_primitives', self.srgb_primitives, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('kernel_type',     self.kernel.type,     mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('hide_emitters',   self.hide_emitters,   mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        if 'kernel_type' in keys:
            self.kernel = Kernel.factory({
                'kernel_type': self.kernel.type,
                'kernel_full_range': True,
                'kernel_normalized': True
            })

    def eval_transmission(self, si, ray, active):
        '''
        Evaluate the transmission model on intersected volumetric primitives
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)
        opacity = si.shape.eval_attribute_1('opacities', si, active)

        # Gaussian splatting transmittance model
        # Find the peak location along the ray. From "3D Gaussian Ray Tracing"
        o = ellipsoid.rot.T * (ray.o - ellipsoid.center) / ellipsoid.scale
        d = ellipsoid.rot.T * ray.d / ellipsoid.scale
        t_peak = -dr.dot(o, d) / dr.dot(d, d)
        p_peak = ray(t_peak)

        density = self.kernel.eval(p_peak, ellipsoid, active)
        transmission = (1.0 - dr.minimum(opacity * density, 0.9999))

        return transmission

    def eval_sh_emission(self, si, ray, active):
        '''
        Evaluate the SH directionally emission on intersected volumetric primitives
        '''
        def eval(shape, si, ray, active):
            if shape is not None and shape.shape_type() == +mi.ShapeType.Ellipsoids:
                sh_coeffs = shape.eval_attribute_x("sh_coeffs", si, active)
                sh_degree = int(dr.sqrt((sh_coeffs.shape[0] // 3) - 1))
                sh_dir_coef = dr.sh_eval(ray.d, sh_degree)
                emission = mi.Color3f(0.0)
                for i, sh in enumerate(sh_dir_coef):
                    emission += sh * mi.Color3f(
                        [sh_coeffs[i * 3 + j] for j in range(3)]
                    )
                return dr.maximum(emission + 0.5, 0.0)
            else:
                return mi.Color3f(0.0)

        return dr.dispatch(si.shape, eval, si, ray, active)

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # --------------------- Configure integrator state ---------------------

        primal = mode == dr.ADMode.Primal
        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        if not primal:  # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)  # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0)  # Differential radiance
        β  = mi.Spectrum(1.0) # Path throughput weight

        # ----------------------------- Main loop ------------------------------

        while dr.hint(active, label=f"Primitive splatting ({mode.name})"):

            # --------------------- Find next intersection ---------------------

            si = scene.ray_intersect(
                ray,
                coherent=(depth == 0),
                ray_flags=mi.RayFlags.All | mi.RayFlags.BackfaceCulling,
                active=active,
            )

            active &= si.is_valid() & (si.shape.shape_type() == +mi.ShapeType.Ellipsoids)

            # ----------------- Primitive emission evaluation ------------------

            Le = mi.Spectrum(0.0)

            with dr.resume_grad(when=not primal):
                emission     = self.eval_sh_emission(si, ray, active)
                transmission = self.eval_transmission(si, ray, active)
                Le[active] = β * (1.0 - transmission) * emission
                Le[~dr.isfinite(Le)] = 0.0

            # ------- Update loop variables based on current interaction -------

            L[active] = (L + Le) if primal else (L - Le)
            β[active] *= transmission

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[active] = si.p + ray.d * 1e-4

            # -------------- Differential phase only (PRB logic) ---------------

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Differentiable reflected indirect radiance for primitives
                    Lr_ind = L * transmission / dr.detach(transmission)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_ind
                    Lo = dr.select(active & dr.isfinite(Lo), Lo, 0.0)

                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # ----------------------- Stopping criterion -----------------------

            active &= si.is_valid()
            depth[active] += 1

            # Kill path if has insignificant contribution
            β_max = dr.max(β)
            active &= (β_max > 0.01)

            # Perform Russian Roulette
            sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            if primal and self.use_rr:
                rr_prob = dr.maximum(β_max, 0.1)
                rr_active = (depth >= self.rr_depth) & (β_max < 0.1)
                β[rr_active] *= dr.rcp(rr_prob)
                rr_continue = sample_rr < rr_prob
                active &= ~rr_active | rr_continue

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

        # Convert sRGB light transport to linear color space
        if self.srgb_primitives:
            L = mi.math.srgb_to_linear(L)

        return L if primal else δL, True, [], L

    def to_string(self):
        return f"VolumetricPrimitiveRadianceFieldIntegrator[]"

mi.register_integrator("volprim_rf", lambda props: VolumetricPrimitiveRadianceFieldIntegrator(props))
