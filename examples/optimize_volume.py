# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
This script implements a simple optimization pipeline that reconstruct an
absorbing volume using volumetric primitives (no scattering!).
'''

import os, argparse
from os.path import join

import numpy as np

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
T = mi.ScalarTransform4f

import volprim
from volprim.integrators.common import *

# Script arguments -------------------------------------------------------------

parser = argparse.ArgumentParser(description='Optimize volumetric primitives from 3D grid')

# Required arguments
parser.add_argument('--output',      type=str, required=True, help='Path to the result output folder')
parser.add_argument('--volume_grid', type=str, required=True, help='Path to .vol volume grid file')

# Cameras and reference images parameters
parser.add_argument('--cam_count',      type=int, default=8,      help='Number of cameras to use in the optimization process')
parser.add_argument('--cam_res',        type=int, default=256,    help='Camera resolution')

# Integrator parameters
parser.add_argument('--ref_spp',   type=int, default=128,         help='Number of samples per pixel used for rendering the reference images')
parser.add_argument('--opt_spp',   type=int, default=1,          help='Number of samples per pixel used for rendering during the optimization')
parser.add_argument('--grad_spp',  type=int, default=1,          help='Number of samples per pixel used for gradient computation during the optimization')
parser.add_argument('--max_depth', type=int, default=-1,         help='Maximum path depth for integrator')
parser.add_argument('--kernel',    type=str, default='gaussian', help='Kernel type')

# Optimization parameters
parser.add_argument('--iterations',        type=int,   default=512,      help='Number of iterations for the gradient descent optimization')
parser.add_argument('--volprim_count',     type=int,   default=16,      help='Resolution of the 3D grid of volumetric primitives')
parser.add_argument('--init_albedo',       type=float, default=0.0,     help='Initial value for the volumetric primitive albedo')
parser.add_argument('--init_sigmat',       type=float, default=0.0001,  help='Initial value for the volumetric primitive sigmat')
parser.add_argument('--no_prune',          action="store_true",         help='Whether insignificant volumetric primitives should be pruned after the optimization')
parser.add_argument('--write_image_every', type=int,   default=4,       help='Frequency as which images and stats are computed')
parser.add_argument('--global_lr',         type=float, default=1.0,     help='Global learning rate')
parser.add_argument('--centers_lr',        type=float, default=0.015,   help='Learning rate multiplier for the volumetric primitive center parameters')
parser.add_argument('--scales_lr',         type=float, default=0.0001,  help='Learning rate multiplier for the volumetric primitive scale parameters')
parser.add_argument('--quats_lr',          type=float, default=0.0001,  help='Learning rate multiplier for the volumetric primitive quat parameters')
parser.add_argument('--sigmat_lr',         type=float, default=0.0001,  help='Learning rate multiplier for the volumetric primitive sigmat parameters')
parser.add_argument('--albedo_lr',         type=float, default=0.0,     help='Learning rate multiplier for the volumetric primitive albedo parameters')
args = parser.parse_args()

# Prepare output folder
os.makedirs(args.output, exist_ok=True)
os.makedirs(join(args.output, 'frames'), exist_ok=True)
os.makedirs(join(args.output, 'refs'),   exist_ok=True)

# Prepare cameras --------------------------------------------------------------

np.random.seed(0)

cameras = []
for i in range(args.cam_count):
    angle = 180.0 / args.cam_count * i - 90.0
    sensor_rotation_y = T().rotate([0, 1, 0], angle)
    sensor_rotation_x = T().rotate([1, 0, 0], 90.0 * np.random.rand() - 45.0) # Randomize camera's elevation a bit
    sensor_to_world = T().look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
    sensor_to_world = sensor_rotation_y @ sensor_rotation_x @ sensor_to_world
    d = {
        'type': 'perspective',
        'fov': 40,
        'to_world': sensor_to_world,
        'film': {
            'type': 'hdrfilm',
            'width': args.cam_res,
            'height': args.cam_res,
            'filter': { 'type': 'gaussian' }
        }
    }
    cameras.append(mi.load_dict(d))

# Render reference images ----------------------------------------

scene_ref = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'prbvolpath',
    },
    'object': {
        'type': 'cube',
        'bsdf': { 'type': 'null' },
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'filename': args.volume_grid,
                'to_world': T().rotate([1, 0, 0], -0).scale([1, 2, 1]).translate(-0.5)
            },
            'albedo': args.init_albedo,
            'scale': 5
        }
    },
    'environment': { 'type': 'constant' }
})

print('Rendering reference images:')
ref_images = []
for i in range(args.cam_count):
    img = mi.render(scene_ref, sensor=cameras[i], spp=args.ref_spp)
    mi.util.write_bitmap(join(args.output, 'refs', f'{i:04d}.exr'), img)
    ref_images.append(img)
    print(f'-- {i+1}/{args.cam_count}', end='\r')
print()

ref_images = [dr.clip(image, 0.0, 1.0) for image in ref_images]

del scene_ref

# Initialize volumetric primitive scene ----------------------------------------

from volprim.integrators.common import EllipsoidsFactory

# Initialize scene with a single Gaussian
factory = EllipsoidsFactory()

delta = 1.0 / args.volprim_count
for x in range(args.volprim_count):
    for y in range(args.volprim_count):
        for z in range(args.volprim_count):
            center = 2.0 * delta * mi.ScalarVector3f(x, y, z) - 1
            factory.add(mean=center, scale=delta / 2, sigmat=args.init_sigmat, albedo=args.init_albedo)

centers, scales, quaternions, sigmats, albedos = factory.build()

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'volprim_tomography',
        'max_depth': args.max_depth,
    },
    'primitives': {
        'type': 'ellipsoidsmesh',
        'centers': centers,
        'scales': scales,
        'quaternions': quaternions,
        'sigma_t': sigmats,
        'albedo': albedos,
        'extent': 3.0,
    },
    'environment': { 'type': 'constant' }
}
for i in range(args.cam_count):
    scene_dict[f'cam_{i:04d}'] = cameras[i]

scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)

# Create batch sensor ----------------------------------------------------------

batch_sensor_dict = {
    'type': 'batch',
    'film': {
        'type': 'hdrfilm',
        'width': args.cam_res * args.cam_count, 'height': args.cam_res,
        'filter': { 'type': 'tent' },
    }
}
for i in range(args.cam_count):
    batch_sensor_dict[f'cam_{i:04d}'] = cameras[i]
batch_sensor = mi.load_dict(batch_sensor_dict)

ref_image = volprim.utils.concatenate_tensors(ref_images)
print(f"Writing image to '{join(args.output, 'reference.exr')}'...")
mi.util.write_bitmap(join(args.output, 'reference.exr'), ref_image)

# Render initial scene ---------------------------------------------------------

with volprim.utils.time_operation('Render initial images'):
    init_image = mi.render(scene, sensor=batch_sensor, spp=args.ref_spp)
print(f"Writing image to '{join(args.output, 'initial.exr')}'...")
mi.util.write_bitmap(join(args.output, 'initial.exr'), init_image)

# Setup optimization -----------------------------------------------------------

key_data   = 'primitives.data'
key_sigmat = 'primitives.sigma_t'
key_albedo = 'primitives.albedo'

opt = volprim.optimizers.BoundedAdam()

ellipsoids = Ellipsoid.unravel(params[key_data])
opt['centers'] = ellipsoids.center
opt['scales']  = ellipsoids.scale
opt['quats']   = mi.Vector4f(ellipsoids.quat)
opt['sigmat']  = params[key_sigmat]
opt['albedo']  = params[key_albedo]

opt.set_learning_rate({
    'centers': args.global_lr * args.centers_lr,
    'scales':  args.global_lr * args.scales_lr,
    'quats':   args.global_lr * args.quats_lr,
    'sigmat':  args.global_lr * args.sigmat_lr,
    'albedo':  args.global_lr * args.albedo_lr,
})

opt.set_bounds('scales', lower=1e-6)
opt.set_bounds('sigmat', lower=1e-8, upper=1e-3)
opt.set_bounds('albedo', lower=1e-8, upper=1.0)

def update_params(opt):
    params[key_data] = Ellipsoid.ravel(opt['centers'], opt['scales'], mi.Quaternion4f(opt['quats']))
    params[key_sigmat] = opt['sigmat']
    params[key_albedo] = opt['albedo']
    params.update()

update_params(opt)

# Optimize! --------------------------------------------------------------------

loss_list = []
psnr_list = []

print(f'Run optimization:')

for it in range(args.iterations):
    image = mi.render(scene, params, sensor=batch_sensor, spp=args.opt_spp, spp_grad=args.grad_spp, seed=it)

    loss = volprim.optimizers.l1(ref_image, image)
    psnr = volprim.optimizers.psnr(ref_image, image)

    dr.backward(loss)
    opt.step()
    update_params(opt)

    loss_list.append(loss.array[0])
    psnr_list.append(psnr.array[0])

    with dr.suspend_grad():
        if (it + 1) % args.write_image_every == 0:
            mi.util.write_bitmap(join(args.output, 'frames', f'image_{it:04d}.exr'), image)

    print(f'-- step {it+1} / {args.iterations} | psnr={psnr_list[-1]:.04f} | loss={loss_list[-1]:.04f}', end='\r')

print()
print('Done with optimization')

# Prune volumetric primitives based on sigma_t values --------------------------

if not args.no_prune:
    valid = (opt['sigmat'] > 1e-6) & dr.all(opt['scales'] > 1e-4) # Pruning threshold based on density and scale
    valid_indices = dr.compress(valid)

    print(f"Pruning {dr.width(opt['sigmat']) - dr.width(valid_indices)} volumetric primitives out of {dr.width(opt['sigmat'])}")
    print(f"--> {dr.width(valid_indices)} volumetric primitives left")

    opt['centers'] = dr.gather(mi.Point3f,  opt['centers'], valid_indices)
    opt['scales']  = dr.gather(mi.Vector3f, opt['scales'],  valid_indices)
    opt['sigmat']  = dr.gather(mi.Float,    opt['sigmat'],  valid_indices)
    opt['albedo']  = dr.ravel(dr.gather(mi.Color3f,  opt['albedo'],  valid_indices))
    opt['quats']   = mi.Quaternion4f(dr.gather(mi.Vector4f, opt['quats'], valid_indices))

    update_params(opt)

# Save results -----------------------------------------------------------------

optimized = mi.render(scene, sensor=batch_sensor, spp=args.ref_spp)
print(f"Writing image to '{join(args.output, 'optimized.exr')}'...")
mi.util.write_bitmap(join(args.output, 'optimized.exr'), optimized)

volprim.io.dict_to_asset(volprim.io.object_to_dict(scene), join(args.output, 'optimized_asset'))

print(f'PSNR: {volprim.optimizers.psnr(ref_image, optimized).array[0]}')

def plot_loss(data, label, output_file):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(data)
    fig.canvas.toolbar_visible = 'fade-in-fade-out'
    fig.canvas.footer_visible = False
    fig.canvas.header_visible = False
    ax.set_xlabel('Iteration')
    plt.ylabel(label)
    plt.title(label + ' plot')
    plt.savefig(output_file)

plot_loss(loss_list, label='Loss', output_file=join(args.output, 'loss.png'))
plot_loss(psnr_list, label='PSNR', output_file=join(args.output, 'psnr.png'))
