# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator, mis_weight
from .common import *
from .stack import alloc_stack

class VolumetricPrimitivesPRBIntegrator(RBIntegrator):
    '''
    This plugin implements the volumetric-primitives path tracer integrator an
    off-the-shelf volumetric path tracer supporting next-event estimation, with
    the key modifications of the transmittance evaluation and sampling, and the
    medium interaction routines to work with volumetric primitives.

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        rr_depth (int): Minimum path depth before enabling the Russian roulette path termination.
        use_nee (bool): Use next-event estimation for indirect light sampling.
        use_indirect (bool): Use indirect light sampling.
        hide_emitters (bool): Hide emitters from indirect light sampling.
        phasefunction (dict): Phase function to use for volumetric interactions.
        kernel_type (str): Name of the kernel to use for rendering the volumetric primitives, one of ['gaussian', 'epanechnikov'].
        max_overlaps (int): Maximum number of overlapping primitives.
        max_depth_primitive (int): Maximum depth for primitive ray intersection.
        rr_depth_primitive (int): Minimum depth before enabling Russian roulette for primitive ray intersection.
        solver_max_iterations (int): Maximum number of iterations for the solver.
        solver_type (str): Type of solver to use for the volumetric interactions. One of ['bisection', 'newton', 'disabled']
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", -1))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces

        rr_depth       = int(props.get('rr_depth', -1))
        self.rr_depth  = mi.UInt32(rr_depth if rr_depth > 0 else 2**32-1)
        self.use_rr    = rr_depth < max_depth

        self.use_nee       = bool(props.get("use_nee", True))     # Slow!
        self.use_indirect  = bool(props.get("use_indirect", True))

        self.phase = props.get('phasefunction')
        if self.phase is None:
            self.phase = mi.load_dict({'type': 'isotropic'})

        props['kernel_full_range'] = False
        props['kernel_normalized'] = False
        self.kernel = Kernel.factory(props)

        self.max_overlaps        = int(props.get("max_overlaps", 32))
        self.max_depth_primitive = int(props.get("max_depth_primitive", 256))
        self.rr_depth_primitive  = int(props.get("rr_depth_primitive",  256))

        self.solver_max_iterations = int(props.get('solver_max_iterations', 4))
        self.solver_type = str(props.get('solver_type', 'bisection'))
        assert self.solver_type in ['bisection', 'newton', 'disabled']

    def traverse(self, callback):
        callback.put_parameter("max_depth",  self.max_depth,  mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("rr_depth",   self.rr_depth,   mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        env_emitter = scene.environment()
        if env_emitter is None:
            mi.Log(mi.LogLevel.Warn, 'Environment emitter is required in volprim_rf integrator!')
        if len(scene.emitters()) > 1:
            mi.Log(mi.LogLevel.Warn, 'Only environment emitters are supported in volprim_rf integrator!')

        check_ellipsoids_attribute(scene, ['sigma_t', 'albedo'])

        # ------------------------------------------------------------------
        # Prepare integrator state
        # ------------------------------------------------------------------

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(dr.detach(ray))
        active    = mi.Bool(active)
        valid_ray = mi.Bool(False)
        depth     = mi.UInt32(0)

        prev_event = dr.zeros(mi.Interaction3f)
        prev_event_pdf = mi.Float(1.0)

        bsdf_ctx  = mi.BSDFContext()
        phase_ctx = mi.PhaseFunctionContext(sampler)

        if not primal: # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β  = mi.Spectrum(1.0)

        # Primitives overlapping with the current path vertex
        primitives = alloc_stack(PrimitiveID, mi.UInt32, alloc_size=self.max_overlaps)

        while active:

            # ------------------------------------------------------------------
            # Sample volumetric interaction
            # ------------------------------------------------------------------

            primitives, si, weight, _, _, _, tr_β, sampled_t = primitive_tracing(
                scene, sampler, ray, primitives, depth,
                callback=self.sample_segment,
                payload=(self, scene, sampler.next_1d(), mi.Float(1.0), mi.Float(dr.inf)),
                active=active,
                max_depth_primitive=self.max_depth_primitive,
                rr_depth_primitive=self.rr_depth_primitive
            )

            # Kill this path if it reached the maximum number of primitive
            # ray-intersection before sampling a medium interaction
            β *= weight
            active &= dr.any(β != 0.0)

            mei = dr.zeros(mi.MediumInteraction3f)
            mei.wi = -ray.d
            mei.sh_frame = mi.Frame3f(mei.wi)
            mei.t = sampled_t
            mei.p = ray(sampled_t)

            active_surface = active & ~mei.is_valid() & si.is_valid()
            active_medium  = active &  mei.is_valid()
            escaped_medium = active & ~mei.is_valid() & ~si.is_valid()
            valid_ray |= active_medium | active_surface

            depth[active_surface | active_medium] += 1
            active_medium  &= depth < self.max_depth
            active_surface &= depth < self.max_depth

            # Propagate derivatives for the transmittance term (cancelled out by sampling PDF)
            if not primal:
                ray2 = dr.zeros(mi.Ray3f)
                ray2.o = mi.Point3f(ray.o)
                ray2.d = dr.select(escaped_medium, -mei.wi, dr.normalize(dr.select(active_surface, si.p, mei.p) - ray2.o)) # TODO isn't this just ray.d?
                ray2.maxt = dr.select(escaped_medium, dr.inf, dr.norm(ray2.d)) # TODO isn't the norm just 1??
                active2 = active_medium | active_surface | (escaped_medium & mi.Bool(self.use_indirect))
                self.eval_transmittance(scene, sampler, ray2, primitives, L, δL, mode, active2)

            # ------------------------------------------------------------------
            # Update path throughput based on volumetric interaction albedo
            # ------------------------------------------------------------------

            albedo = self.eval_albedo(scene, mei.p, primitives, L, δL, mode, active_medium)
            β[active_medium] *= albedo

            # ------------------------------------------------------------------
            # Interaction with environment emitter
            # ------------------------------------------------------------------

            if self.use_indirect:
                # Get the PDF of sampling this emitter using next event estimation
                if self.use_nee:
                    ds = mi.DirectionSample3f(scene, si, prev_event)
                    emitter_pdf = env_emitter.pdf_direction(mei, ds, escaped_medium & (depth != 0))
                else:
                    emitter_pdf = mi.Float(0.0)

                with dr.resume_grad(when=not primal):
                    emitter_val = env_emitter.eval(si, escaped_medium)
                    escaped_medium &= ~((depth == 0) & mi.Bool(self.hide_emitters))
                    Lr_dir = β * mis_weight(prev_event_pdf, emitter_pdf) * emitter_val
                    Lr_dir = dr.select(escaped_medium, Lr_dir, 0.0)

                    # Propagate derivatives to/from `emitter_val`
                    if not primal:
                        if dr.grad_enabled(Lr_dir):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_dir)
                            else:
                                δL += dr.forward_to(Lr_dir)

                L = (L + Lr_dir) if primal else (L - Lr_dir)

            # ------------------------------------------------------------------
            # Emitter sampling
            # ------------------------------------------------------------------

            bsdf = si.bsdf(ray)

            if self.use_nee:
                ref = dr.zeros(mi.Interaction3f)
                ref[active_medium]  = mei
                ref[active_surface] = si

                active_surface_nee = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

                with dr.resume_grad(when=not primal):
                    ds, emitter_val = env_emitter.sample_direction(ref, sampler.next_2d(), active_surface_nee | active_medium)
                    ray2 = ref.spawn_ray_to(ds.p)
                    transmittance = self.eval_transmittance(scene, sampler, ray2, primitives, L, δL, dr.ADMode.Primal, active)

                    # Query the BSDF for that emitter-sampled direction
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, si.to_local(ds.d), active_surface_nee)
                    phase_val, phase_pdf = self.phase.eval_pdf(phase_ctx, mei, ds.d, active_medium)
                    phase_pdf = dr.detach(dr.select(ds.delta, 0.0, phase_pdf))
                    if not self.use_indirect:
                        phase_pdf = mi.Float(0.0)

                    nee_val = dr.select(active_surface_nee, bsdf_val, phase_val)
                    nee_pdf = dr.select(ds.delta, 0.0, dr.select(active_surface_nee, bsdf_pdf, phase_pdf))

                    if not self.use_indirect:
                        nee_pdf = mi.Float(0.0)

                    Lr_nee = β * nee_val * mis_weight(ds.pdf, nee_pdf) * transmittance * emitter_val
                    Lr_nee = dr.select(active_surface | active_medium, Lr_nee, 0.0)

                    # Propagate derivatives to/from `emitter_val` and `phase_val`
                    if not primal:
                        if dr.grad_enabled(Lr_nee):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_nee)
                            else:
                                δL += dr.forward_to(Lr_nee)

                # TODO why shouldn't we propagate the gradients of the transmittance term in NEE?
                # if not primal:
                #     self.eval_transmittance(scene, sampler, ray2, primitives, Lr_nee, δL, mode, active_medium)

                L = (L + Lr_nee) if primal else (L - Lr_nee)

            # ------------------------------------------------------------------
            # Phase function sampling
            # ------------------------------------------------------------------

            # TODO support spatially varying phase function

            wo, phase_weight, phase_pdf = self.phase.sample(phase_ctx, mei,
                                                            sampler.next_1d(),
                                                            sampler.next_2d(),
                                                            active_medium)
            active_medium &= phase_pdf > 0.0

            β[active_medium] *= phase_weight

            ray.o[active_medium]    = mei.p
            ray.d[active_medium]    = wo
            ray.maxt[active_medium] = dr.inf

            prev_event[active_medium] = dr.detach(mei)
            prev_event_pdf[active_medium] = phase_pdf

            # ------------------------------------------------------------------
            # BSDF sampling
            # ------------------------------------------------------------------

            bs, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                          sampler.next_1d(active_surface),
                                          sampler.next_2d(active_surface),
                                          active_surface)
            active_surface &= bs.pdf > 0

            if not primal:
                with dr.resume_grad():
                    bsdf_eval = bsdf.eval(bsdf_ctx, si, bs.wo, active_surface)
                    if dr.grad_enabled(bsdf_eval):
                        Lo = bsdf_eval * dr.detach(dr.select(active_surface, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * Lo)
                        else:
                            δL += dr.forward_to(Lo)

            β[active_surface] *= bsdf_weight
            ray[active_surface] = si.spawn_ray(si.to_world(bs.wo))

            prev_event[active_surface] = dr.detach(si)
            prev_event_pdf[active_surface] = bs.pdf

            # ------------------------------------------------------------------
            # Termination criterion (russian roulette)
            # ------------------------------------------------------------------

            sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            if primal and self.use_rr:
                q = dr.minimum(dr.max(β), 0.99)
                perform_rr = (depth > self.rr_depth)
                active &= (sample_rr < q) | ~perform_rr
                β[perform_rr] = β * dr.rcp(q)

            # Kill path if has insignificant contribution
            active &= dr.any(β > 0.005)
            active &= dr.any((β != 0.0))
            active &= active_surface | active_medium


        return L if primal else δL, valid_ray, [], L

    @dr.syntax
    def eval_transmittance(self, scene, sampler, ray, primitives, L, δL, mode, active):
        '''
        Evaluate the transmittance along a ray segment
        '''
        # Early exit if the scene doesn't contains volumetric primitives
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return mi.Float(1.0)

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(ray)
        L = dr.detach(L)

        # ----------------------------------------------------------------------
        # Process overlapping primitives
        # ----------------------------------------------------------------------

        total_tr = mi.Float(1.0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        it = mi.UInt32(0)
        while active1:
            prim_id = primitives.value(it, active1)
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_id.index
                sigma_t = si.shape.eval_attribute_1('sigma_t', si, active1)
                density = self.kernel.density_integral(ray, ellipsoid, tmin=mi.Float(0.0), tmax=ray.maxt, active=active1)

                if False: # TODO why would this be necessary??
                    # Make sure we evaluate the density of the gaussian in the valid range
                    valid, t0, t1 = ray_ellipsoid_intersection(ray, ellipsoid, active1)
                    t0 = mi.Float(0.0)
                    t1 = dr.minimum(t1, ray.maxt)

                tr = dr.exp(-density * sigma_t)

                valid = active1
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)

                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active1 & ((1.0 - tr) > 1e-2)
                    contrib[~valid] = 0.0

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            total_tr[valid] *= tr
            it += 1
            active1 &= (it < primitives.size())

        # ----------------------------------------------------------------------
        # Process remaining ray segment
        # ----------------------------------------------------------------------

        it = mi.UInt32(0)
        active2 = mi.Bool(active)
        while active2:
            # Find the next Gaussian along the ray
            si = scene.ray_intersect(ray, coherent=False, ray_flags=(mi.RayFlags.All | mi.RayFlags.BackfaceCulling), active=active2)
            active2 &= si.is_valid()

            # TODO handle the case where we hit a surface
            # This should never happens as in case a surface is hit through this ray, we will have zero radiance to backpropagate
            # For instance, this function is called after sampling interaction, in which case we will stop before hitting a surface.
            # For the case of NEE (currently this isn't called), L will only contain non-zero radiance when no surface was hit
            # during the primal computation of the path.

            # Handle the case where we hit a surface
            is_ellipsoids = (si.shape.shape_type() == +mi.ShapeType.Ellipsoids)
            hit_surface = si.is_valid() & ~is_ellipsoids
            total_tr[active2 & hit_surface] = 0.0
            active2 &= ~hit_surface

            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active2)
                sigma_t = si.shape.eval_attribute_1('sigma_t', si, active2)
                density = self.kernel.density_integral(ray, ellipsoid, tmin=None, tmax=None, active=active2) # TODO if ray.maxt is not inf (envmap), don't evaluate full range!

                tr = dr.exp(-density * sigma_t)

                valid = active2
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)
                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active2 & ((1.0 - tr) > 1e-2)
                    contrib = dr.select(valid, contrib, 0.0)

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            total_tr[valid] *= tr
            active2 &= (total_tr > 0.001) # TODO this is biased, should do russian roulette instead?

            # Update rays
            ray.o = spawn_ray_origin(si, ray)
            ray.maxt -= si.t

            it += 1

            if primal: # TODO check random number sequence!
                # Perform Russian Roulette
                if self.rr_depth_primitive > 0:
                    q = dr.minimum(total_tr, 0.99)
                    perform_rr = (it > self.rr_depth_primitive) & (q > 0.0)
                    valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
                    total_tr[active2 & perform_rr] *= dr.rcp(q)
                    total_tr[active2 & ~valid_rr] = 0.0
                    active2 &= valid_rr

                # Kill paths that are too long
                if self.max_depth_primitive > 0:
                    total_tr[active2 & (it >= self.max_depth_primitive)] = 0.0
                    active2 &= (it < self.max_depth_primitive)


        return total_tr

    @dr.syntax
    def eval_albedo(self, scene, p, primitives, L, δL, mode, active):
        '''
        Returns the albedo value of the volumetric primitives involved in the
        interaction. This is compute as the weighted sum of the volumetric
        primitives albedos, with the PDF as weights.
        '''
        # Early exit if the scene doesn't contains volumetric primitives
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return dr.zeros(mi.Spectrum)


        accum_albedo = dr.zeros(mi.Spectrum)
        accum_pdf    = dr.zeros(mi.Float)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        it1 = mi.UInt32(0)
        while active1:
            prim_id = primitives.value(it1, active1)
            si1 = dr.zeros(mi.SurfaceInteraction3f)
            si1.prim_index = prim_id.index
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            sigma_t   = prim_id.shape.eval_attribute_1('sigma_t', si1, active1)
            albedo    = prim_id.shape.eval_attribute_3('albedo', si1, active1)
            pdf = self.kernel.pdf(p, ellipsoid, active1)
            pdf *= sigma_t
            accum_pdf += pdf
            accum_albedo += pdf * albedo
            it1 += 1
            active1 &= (it1 < primitives.size())

        # ----------------------------------------------------------------------
        # Adjoint albedo propagation
        # ----------------------------------------------------------------------

        if mode != dr.ADMode.Primal:
            L_albedo = dr.detach(L) * safe_rcp(accum_albedo * safe_rcp(accum_pdf))

            # Propagate derivatives for albedo terms
            active2 = mi.Bool(active) & ~primitives.is_empty()
            it2 = mi.UInt32(0)
            while active2:
                prim_id = primitives.value(it2, active2)

                with dr.resume_grad():
                    si1 = dr.zeros(mi.SurfaceInteraction3f)
                    si1.prim_index = prim_id.index
                    ellipsoid = Ellipsoid.gather(prim_id.shape, si1.prim_index, active2)
                    sigma_t   = prim_id.shape.eval_attribute_1('sigma_t', si1, active2)
                    albedo    = prim_id.shape.eval_attribute_3('albedo', si1, active2)
                    pdf = self.kernel.pdf(p, ellipsoid, active2)
                    pdf *= sigma_t

                    contrib = mi.Spectrum(0.0)

                    # 1. Derivative of the `accum_albedo` term in `accum_albedo / accum_pdf`
                    d_albedo_pdf = dr.replace_grad(1.0, albedo * pdf)
                    contrib += d_albedo_pdf * dr.detach(safe_rcp(accum_pdf) * L_albedo)

                    # 2. Derivative of the `accum_pdf` term in `accum_albedo / accum_pdf`
                    d_pdf = dr.replace_grad(1.0, pdf)
                    contrib += -d_pdf * dr.detach(accum_albedo * safe_rcp(accum_pdf * accum_pdf) * L_albedo)

                    # 3. Derivative of the pdf term that cancels out in the weight (as it is also contained in the PDF)
                    contrib += d_pdf * dr.detach(L * safe_rcp(accum_pdf))

                    contrib = dr.select(active2, contrib, 0.0)

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            δL += dr.forward_to(contrib)

                it2 += 1
                active2 &= (it2 < primitives.size())

        albedo = accum_albedo * safe_rcp(accum_pdf)
        return albedo

    @dr.syntax
    def sample_segment(self, payload, primitives, ray, accum_t, seg_t0, seg_t1, active):
        # Fetch integrator state from payload
        self, scene, sample, β, sampled_t = payload

        # ----------------------------------------------------------------------
        # Compute segment transmittance
        # ----------------------------------------------------------------------


        seg_tau = mi.Float(0.0)
        it  = mi.UInt32(0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        while active1:
            prim_id = primitives.value(it, active1)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            density = self.kernel.density_integral(ray, ellipsoid, seg_t0, seg_t1, active1)
            si1 = dr.zeros(mi.SurfaceInteraction3f)
            si1.prim_index = prim_id.index
            sigma_t = prim_id.shape.eval_attribute_1('sigma_t', si1, active1)
            seg_tau[active1] += density * sigma_t
            it += 1
            active1 &= (it < primitives.size())
        seg_tr = dr.exp(-seg_tau)

        # ----------------------------------------------------------------------
        # Test segment sampling condition
        # ----------------------------------------------------------------------

        chi_i = β * seg_tr
        success = active & (chi_i < sample) & ~primitives.is_empty()

        if self.solver_type == 'disabled':
            # Sample uniformly within the segment (slightly biased but much faster)
            t_s = seg_t0 + sample * (seg_t1 - seg_t0)
        else:
            t_s = self.primitives_sample_interaction_segment(scene, ray, primitives, seg_t0, seg_t1, sample, β, success)

        sampled_t[success] = t_s + accum_t


        β[active] *= seg_tr
        β[success] = mi.Float(sample)

        return ~success, (self, scene, sample, β, sampled_t)

    @dr.syntax
    def primitives_sample_interaction_segment(self, scene, ray, primitives, seg_t0, seg_t1, sample, β, active):
        '''
        Sample a scattering distance for a segment defined by [seg_t0, seg_t1]
        '''
        active = mi.Bool(active) & ~primitives.is_empty()

        # Initial guess for the Newton solver
        t_s = (seg_t0 + seg_t1) / 2.0
        chi = -dr.log(sample / β)
        if self.kernel_type_ == "gaussian":
            # Handle the case where we have a single primitive
            single_case = active & (primitives.size() == 1)
            prim_id = primitives.value(mi.UInt32(0), single_case)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, single_case)
            si1 = dr.zeros(mi.SurfaceInteraction3f)
            si1.prim_index = prim_id.index
            sigma_t = prim_id.shape.eval_attribute_1('sigma_t', si1, single_case)
            t_s = self.kernel.inv_cdf(ray, ellipsoid, sigma_t, chi, single_case)
            success = single_case #& (t_s > seg_t0) & (t_s < seg_t1)
            active &= ~success

        # ----------------------------------------------------------------------
        # Fetching the primitive attributes
        # ----------------------------------------------------------------------

        active1 = mi.Bool(active) & ~primitives.is_empty()
        it1 = mi.UInt32(0)
        ellipsoids = alloc_stack(Ellipsoid, mi.UInt32, alloc_size=primitives.alloc_size())
        sigmats    = alloc_stack(mi.Float,  mi.UInt32, alloc_size=primitives.alloc_size())
        while active1:
            prim_id = primitives.value(it1, active1)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            si1 = dr.zeros(mi.SurfaceInteraction3f)
            si1.prim_index = prim_id.index
            sigma_t = prim_id.shape.eval_attribute_1('sigma_t', si1, active1)

            ellipsoids.push(ellipsoid, active1)
            sigmats.push(sigma_t, active1)

            it1 += 1
            active1 &= (it1 < primitives.size())

        # ----------------------------------------------------------------------
        # Newton / bisection solver
        # ----------------------------------------------------------------------

        # TODO fix Newton solver

        it = mi.UInt32(0)
        while active:
            cdf = mi.Float(0.0)
            pdf = mi.Float(0.0)

            it2  = mi.UInt32(0)
            active2 = mi.Bool(active)
            while active2:
                ellipsoid = ellipsoids.value(it2, active2)
                sigmat    = sigmats.value(it2, active2)

                # Compute integral value
                cdf += self.kernel.density_integral(ray, ellipsoid, seg_t0, t_s, active2) * sigmat 

                if self.solver_type == 'newton':
                    pdf += self.kernel.pdf(ray(t_s), ellipsoid, active2) * sigmat

                it2 += 1
                active2 &= (it2 < ellipsoids.size())

            step = mi.Float(0.0)
            if self.solver_type == 'newton':
                # Newton algorithm
                step = (dr.exp(-chi) - dr.exp(-cdf)) / pdf
                t_s[active] = t_s - step
            else:
                # Bisection algorithm
                step = (seg_t1 - seg_t0) / dr.power(2.0, it + 2)
                t_s[active] = dr.select(cdf > chi, t_s - step, t_s + step)

            # Avoid NaNs and accelerate solver by clipping steps outside of segment
            diverge = active & ((t_s < seg_t0) | (t_s > seg_t1))
            t_s[diverge] = dr.clip(t_s, seg_t0, seg_t1)

            # If diverged at the last step, return t_s as the center of the segment
            t_s[diverge & ((it + 1 == self.solver_max_iterations))] = 0.5 * (seg_t0 + seg_t1)

            it += 1
            active &= (it < self.solver_max_iterations)
            active &= (dr.abs(step) > 5e-6) # TODO select this value according to something (i.e minimum intersection displacement? can possibly be bigger, this is very safe)

        return t_s

    def to_string(self):
        return f'VolumetricPrimitivesPRBIntegrator[]'

mi.register_integrator("volprim_prb", lambda props: VolumetricPrimitivesPRBIntegrator(props))
