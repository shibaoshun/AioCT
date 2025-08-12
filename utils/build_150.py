# import odl
# import numpy as np
#
#
# # 640geo
# class initialization150:
#     def __init__(self):
#         self.param = {}
#         self.reso = 512 / 416 * 0.03
#
#         # image
#         self.param['nx_h'] = 416
#         self.param['ny_h'] = 416
#         self.param['sx'] = self.param['nx_h']*self.reso
#         self.param['sy'] = self.param['ny_h']*self.reso
#
#         ## view
#         self.param['startangle'] = 0
#         self.param['endangle'] = 5/6 * np.pi
#
#         self.param['nProj'] = 150
#
#         ## detector
#         self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
#         self.param['nu_h'] = 641
#         self.param['dde'] = 1075 * self.reso      ########gai 1075
#         self.param['dso'] = 1075 *self.reso     ########gai 1075
#
#         self.param['u_water'] = 0.192
#
#
#
#
# def build_gemotry150(param):
#     reco_space_h = odl.uniform_discr(
#         min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
#         max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
#         dtype='float32')#返回一个均匀离散化的 L^p 函数空间。
#
#     angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
#                                             param.param['nProj'])#######创建分区
#
#     detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
#                                                  param.param['nu_h'])
#
#     geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
#                                           src_radius=param.param['dso'],
#                                           det_radius=param.param['dde'])
#
#     ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
#     FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Ram-Lak', frequency_scaling=1.0)
#     return ray_trafo_hh, FBPOper_hh
#
import odl
import numpy as np

from odl.contrib import torch as odl_torch
import torch
## 640geo
class initialization150:
    def __init__(self):
        self.param = {}
        self.reso = 512 / 416 * 0.03

        # image
        self.param['nx_h'] = 416
        self.param['ny_h'] = 416
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view 0-360 640均分
        self.param['startangle'] = 0
        self.param['endangle'] = 5/6* np.pi

        self.param['nProj'] = 150

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = 641
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso

        self.param['u_water'] = 0.192


def build_gemotry150(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    op_norm = odl.operator.power_method_opnorm(ray_trafo)
    op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()
    op_layer_adjoint = odl_torch.operator.OperatorModule(ray_trafo.adjoint)

    bp = odl_torch.OperatorModule(ray_trafo)
    op_fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0) #* np.sqrt(2)
    fbp = odl_torch.operator.OperatorModule(op_fbp)
    return bp,op_layer_adjoint,fbp,op_norm
