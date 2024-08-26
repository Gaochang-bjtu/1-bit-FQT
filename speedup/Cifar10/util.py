from typing import Any
import binop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd.function import InplaceFunction
import time
from preconditioner import ScalarPreconditioner, DiagonalPreconditioner
import math
#64
import log
import os
#logger = log.get_logger('1_logger.log')
class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 4
        self.bweight_num_bits = 4
        self.backward_persample = True
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True

    def activation_preconditioner(self):
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        #return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        return lambda x: DiagonalPreconditioner(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self):
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.backward_num_bits, False, left=True)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self):
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.bweight_num_bits, True, left=True)
        else:
            return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)

config = QuantizationConfig()

class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=True, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output, zero_point, scale = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(0.0, preconditioner.num_bins).round_()
        return output, zero_point, scale

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None
def bin_save_state(args, model):
    print('==> Binarizing and Saving model ...')
    state = model.state_dict()
    weight_ = []
    for key in state.keys():
        if 'weight' in key and 'bn' not in key:
            weight_.append((key, state.get(key)))

    # except the first and last layer
    weight_.pop(0)
    weight_.pop()

    for key, weight in weight_:
        s = weight.size()
        if len(s) == 4:
            weight = weight.view(s[0], s[1] * s[2] * s[3])

        if args.cuda:
            bin_weight = torch.cuda.IntTensor()
            binop.encode_rows(weight, bin_weight)
        else:
            bin_weight = torch.IntTensor()
            binop.encode_rows_cpu(weight, bin_weight)

        state[key] = bin_weight
    torch.save(state, 'models/' + args.arch + '.pth')


def bin_conv2d(input, weight, bias, alpha, kernel_size, stride, padding):
    output = Conv2d_().apply(input, weight, alpha, config, kernel_size, stride, padding)
    output.data.mul_(alpha.data.t().expand(output.shape))
    
    if bias is not None:
        #print(bias.data.shape, output.shape)
        bias = bias.view(1,-1,1,1)
        output.data.add_(bias.data.expand(output.shape))

    return output
def mask_trans(x):
    result = []
    
    for i in range(x.shape[0]):
        if x[i] == True:
            result.append(1)
        else:
            result.append(0)
    #print(result)
    return result
class Conv2d_(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor, cofig_: QuantizationConfig, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
        start_time = time.time()
        N,C,H,W = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        ctx.C = C
        H_,W_ = int((H + 2 * padding[0] - kernel_size[0] + 1) / stride[0]), int((W + 2 * padding[1] - kernel_size[1] + 1) / stride[1])
        T = H_ *  W_
        input = torch.nn.functional.pad(input, (1,1,1,1),mode='constant', value=0)
        input_unfolder = torch.zeros(N*T, kernel_size[0] * kernel_size[1] * C)
        time1 = time.time()
        #unfolder(input, input_unfolder, kernel_size, N, T, H_, stride)
        
        binop.unfolder(input, input_unfolder, kernel_size, N, T, H_, C, stride[0])
        #exit(0)
        time2 = time.time()
        #print('unfolder_time:', middle_time - start_time)
        weight_unfolder = weight.view(weight.shape[0], -1)   
        ctx.config = cofig_
        m = input_unfolder.data.shape[0]
        n = input_unfolder.data.shape[1]
        k = weight_unfolder.data.shape[0]     
        bin_input = torch.LongTensor()
        bin_weight = torch.LongTensor()
        out_tensor = torch.FloatTensor()
        output = Variable(out_tensor, requires_grad = True)
        time3 = time.time()
        #print(input_unfolder.shape, weight_unfolder.shape)
        binop.encode_rows_cpu(input_unfolder.data, bin_input)
        
        binop.encode_rows_cpu(weight_unfolder.data, bin_weight)
        #print(input_unfolder.shape, weight_unfolder.shape)
        time4 = time.time()
        #time_1 = time.time()
        #binop.full_mm(input_unfolder.data, weight_unfolder.data, output, m, n, k)
        binop.binary_gemm_cpu(bin_input, bin_weight, output.data, m, n, k, 1, 0, 0, 1)

        
        #print(output.shape)
        #time_1_1 = time.time()
        #print(bin_input.shape, bin_weight.shape)
        #print(time_1_1 - time_1)
        #print(bin_input.shape, bin_weight.shape, m, n, k)
        time5 = time.time()
        ctx.save_for_backward(input_unfolder, weight_unfolder)
        ctx.input_unfolder = input_unfolder
        ctx.padding = padding
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        ctx.weight_unfolder = weight_unfolder
        #print(weight_unfolder.shape)
        output.data = output.data.view(N, H_, W_, output.data.shape[-1])
        output.data = output.data.permute(0, 3, 1, 2)
        end_time = time.time()

        #logger.info('unfolder_time: {:.3f} encode_time: {:.3f} compute_time: {:.3f} total_time: {:.3f}'.format(time2 - time1, time4 - time3, time5 - time4, end_time - start_time))
        #print('unfolder_time:', time2 - time1, 'encode_time:', time4 - time3, 'compute_time:', time5 - time4)
        return output 
    @staticmethod              
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        
        total_start_time = time.time()
        cofig = ctx.config
        bin_grad_1 = torch.LongTensor()
        bin_grad_2 = torch.LongTensor()
        bin_grad_3 = torch.LongTensor()
        bin_grad_4 = torch.LongTensor()
        bin_input = torch.LongTensor()
        grad_a_tensor1 = torch.FloatTensor()

        grad_w_tensor1 = torch.FloatTensor()
        alpha = torch.FloatTensor()
        weight_sum = torch.FloatTensor()
        input_sum = torch.FloatTensor()
        bin_weight = torch.LongTensor()
        bin_grad1 = torch.LongTensor()
        bin_grad2 = torch.LongTensor()
        bin_grad3 = torch.LongTensor()
        bin_grad4 = torch.LongTensor()    
        grad_a1 = Variable(grad_a_tensor1, requires_grad = False)
        grad_w1 = Variable(grad_w_tensor1, requires_grad = False)
        #print('1')
        input_unfolder, weight_unfolder = ctx.input_unfolder, ctx.weight_unfolder
        N, T, D = grad_output.shape[0], input_unfolder.shape[0] / grad_output.shape[0], grad_output.shape[1]
        time1 = time.time()
        mask = _mask(grad_output)
        mask1 = _mask(grad_output, False)
        #print(grad_output.shape)
        grad_output1 = grad_output[:,mask1]
        grad_output = grad_output[mask]
        time2 = time.time()
        input = input_unfolder.view(N, -1, input_unfolder.shape[-1])
        input = input.view(-1, input_unfolder.shape[-1])
        time3 = time.time()
        grad_input_weight, zero_point1, scale1= UniformQuantize().apply(grad_output1, cofig.weight_gradient_preconditioner(), True, False)
        grad_input_activation, zero_point2, scale2 = UniformQuantize().apply(grad_output, cofig.activation_gradient_preconditioner(), True, False)

        time4 = time.time()
        #print('2')
        #print(grad_output.shape, grad_input_weight.shape, grad_input_activation.shape)
        time5 = time.time()
        binop.encode_cols_cpu(weight_unfolder, bin_weight)
        binop.encode_cols_cpu(input, bin_input) 
        #print(weight_unfolder.shape)
        #print(weight_unfolder.shape, input.shape, input_unfolder.shape, 'weight and input')    
        #print(bin_weight.shape, bin_input.shape)
        
        grad_input_activation.data = grad_input_activation.data.view(-1,D)
        #print('2_1', grad_input_activation.shape)
        binop.encode_rows_cpu_4(grad_input_activation, bin_grad_1, bin_grad_2, bin_grad_3, bin_grad_4)
        #print('2_2', grad_input_weight.shape)
        #print(bin_grad_1.shape, bin_weight.shape, 1)
        binop.encode_rows_cpu_4(grad_input_weight, bin_grad1, bin_grad2, bin_grad3, bin_grad4) 

        #print('3')
        time6 = time.time()
        #binop.binary_gemm_cpu(bin_grad_1, bin_weight.t(), grad_a1, bin_grad_1.shape[0], grad_input_activation.shape[1], bin_weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_2, bin_weight.t(), grad_a2, bin_grad_1.shape[0], grad_input_activation.shape[1], bin_weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_3, bin_weight.t(), grad_a3, bin_grad_1.shape[0], grad_input_activation.shape[1], bin_weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_4, bin_weight.t(), grad_a4, bin_grad_1.shape[0], grad_input_activation.shape[1], bin_weight.shape[1], 1, 0, 0, alpha)
        #time_1 = time.time()
        #
        binop.binary_gemm_cpu_4_activation(bin_grad_1, bin_grad_2, bin_grad_3, bin_grad_4, bin_weight.t(), grad_a1, bin_grad_1.shape[0], grad_input_activation.shape[1], bin_weight.shape[1], 1, 0, 0, alpha)
        #time_1_1 = time.time()
        #print(time_1_1 - time_1)
        #print(bin_grad_1.shape, bin_weight.t().shape, grad_a1.shape)
        #binop.binary_gemm_cpu(bin_grad1, bin_input.t(), grad_w1, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad2, bin_input.t(), grad_w2, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad3, bin_input.t(), grad_w3, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        
        #binop.binary_gemm_cpu(bin_grad4, bin_input.t(), grad_w4, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #time_2 = time.time()
        binop.binary_gemm_cpu_4_activation(bin_grad1, bin_grad2, bin_grad3, bin_grad4, bin_input.t(), grad_w1, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)

        #time_2_2 = time.time()   
        #print(time_2_2 - time_2)    

        time7 = time.time()
        
        #print('4')
        #print(bin_grad1.shape, bin_input.shape, grad_w1.shape)
        time8 = time.time()
        weight_unfolder = weight_unfolder.to(torch.float)

        bin_weight_sum = weight_unfolder.sum(0).unsqueeze(0)

        zero_point2 = zero_point2.expand(zero_point2.shape[0], T).contiguous().view(-1,1)
        #binop.mm(zero_point2, bin_weight_sum, weight_sum, zero_point2.shape[0], zero_point2.shape[1], bin_weight_sum.shape[1])  
        weight_sum = torch.mm(zero_point2, bin_weight_sum)

        #grad_a1.data = grad_a1.data * scale2.data.expand(scale2.shape[0], T).contiguous().view(-1,1)
        #grad_a1.data = grad_a1.data + weight_sum.data.expand_as(grad_a1.data)
        #print(grad_a1.shape, scale2.data.expand(scale2.shape[0], T).contiguous().view(-1,1).shape, weight_sum.data.expand_as(grad_a1.data).shape)
        binop.dot_add(grad_a1.data, scale2.data.expand(scale2.shape[0], T).contiguous().view(-1,1).data, weight_sum.data.expand_as(grad_a1.data).data, grad_a1.shape[0], grad_a1.shape[1])
        bin_input_sum = input.sum(0).unsqueeze(0)
        #print(zero_point1.shape, bin_input_sum.shape)
        input_sum = torch.mm(zero_point1, bin_input_sum)   
         
        #binop.mm(zero_point1, bin_input_sum, input_sum, zero_point1.shape[0], zero_point1.shape[1], bin_input_sum.shape[1])     

        #grad_w1.data = grad_w1.data * scale1.data
        #print(grad_w1.shape, scale1.shape, input_sum.shape)
        #grad_w1.data = grad_w1.data + input_sum.data 

        binop.dot_add(grad_w1.data, scale1.data, input_sum.data, grad_w1.shape[0], grad_w1.shape[1])
        time9 = time.time()
        #print('5')
        #print(grad_a1.shape, grad_w1.shape, 6)
        k = int(math.sqrt(T))
        H_p, W_p = (k - 1)* ctx.stride[0] + ctx.kernel_size[0], (k - 1)* ctx.stride[1] + ctx.kernel_size[1]
        grad_a1_folder = torch.zeros(N, ctx.C, H_p, W_p)
        mask = mask_trans(mask)
        start_time = time.time()
        binop.folder(grad_a1, grad_a1_folder, mask, ctx.C, N, int(T), int(k), int(H_p), ctx.kernel_size[0], ctx.padding[0], ctx.stride[0])
        #folder(grad_a1, grad_a1_folder, ctx.kernel_size, T, k, mask, ctx.C, ctx.stride)
        end_time = time.time()
        final_start_time = time.time()
        grad_w1_ = torch.zeros(grad_w1.shape[0]*4, grad_w1.shape[1])
        grad_w1_[mask1] = grad_w1
        grad_w1 = grad_w1_.reshape(grad_w1_.shape[0],-1,ctx.kernel_size[0], ctx.kernel_size[1])
        #logger.info("lss_time: {:.3f} quan_time: {:.3f} encode_time: {:.3f} compute_time: {:.3f} dequan_time: {:.3f} fold_time: {:.3f} total_time: {:.3f}".format(time2 - time1, time4 - time3, time6 - time5, time7 - time6, time9 - time8, end_time - start_time, final_start_time - total_start_time))
        #print('lss_time:', time2 - time1, 'quan_time:', time4 - time3, 'encode_time:', time6 - time5, 'compute_time', time7 - time6, 'dequan_time:', time9 - time8, 'folder_time:', end_time - start_time)
        if ctx.stride[0] == 1:
            
            return grad_a1_folder[:, :, ctx.padding[0]:-ctx.padding[0], ctx.padding[1]: -ctx.padding[1]], grad_w1, None, None, None, None, None
        else:
            return grad_a1_folder[:, :, ctx.padding[0]:, ctx.padding[1]:], grad_w1, None, None, None, None, None

def bin_linear(input, weight, bias, alpha):
    m = input.data.shape[0]
    n = input.data.shape[1]
    k = weight.data.shape[0]
    out_tensor = torch.FloatTensor()
    bin_input = torch.LongTensor()
    use_cuda = input.is_cuda
    #output = torch.Float
    if use_cuda:
        bin_input = bin_input.cuda()
        out_tensor = out_tensor.cuda()

    #output = Variable(out_tensor, requires_grad=False)
    output = torch.FloatTensor()
    if use_cuda:
        binop.encode_rows(input.data, bin_input)
        binop.binary_gemm(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
    else:
        #binop.encode_rows_cpu(input.data, bin_input)
        #binop.binary_gemm_cpu(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
        #output = F.linear(input, weight)
        #output = torch.mm(input, torch.t(weight))
        #print(input.shape, weight.shape)
        binop.full_mm(input, weight, output.data, input.shape[0], input.shape[1], weight.shape[0])
        print(input.shape, weight.shape, output.shape)
    output.data.mul_(alpha.data.t().expand(output.shape))
    if bias is not None:
        output.data.add_(bias.data.expand(output.shape))
    return output
def _mask(x, transpose = True):
    if x.dim() == 4:
        if transpose == True:
            abs_value = x.abs().max(-1)[0].max(-1)[0].max(-1)[0]
            th = torch.topk(abs_value, int(abs_value.shape[0] / 4))[0][-1]
            mask = torch.where(abs_value>=th, torch.tensor(True), torch.tensor(False))
            mask = [True] * int(abs_value.shape[0] / 4) + [False] * (abs_value.shape[0] - int(abs_value.shape[0] / 4))
            mask = torch.tensor(mask)
            return mask
        else:
            abs_value = x.abs().max(-1)[0].max(-1)[0].max(0)[0]
            th = torch.topk(abs_value, int(abs_value.shape[0] / 4))[0][-1]
            mask = torch.where(abs_value>=th, torch.tensor(True), torch.tensor(False))
            mask = [True] * int(abs_value.shape[0] / 4) + [False] * (abs_value.shape[0] - int(abs_value.shape[0] / 4))
            mask = torch.tensor(mask)
            return mask
        #mask = mask.reshape(-1, 1, 1, 1)
    elif x.dim() == 2:
        if transpose == True:
            abs_value = x.abs().max(-1)[0]
        else:
            abs_value = x.abs().max(0)[0]
        th = torch.topk(abs_value, int(abs_value.shape[0] / 4))[0][-1]
        mask = torch.where(abs_value>=th, torch.tensor(True), torch.tensor(False))
        #print(abs_value, th)
        #mask = mask.reshape(-1, 1)   
        mask = [True] * int(abs_value.shape[0] / 4) + [False] * (abs_value.shape[0] - int(abs_value.shape[0] / 4))
        mask = torch.tensor(mask)
        return mask
class Linear_(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor, cofig: QuantizationConfig) -> torch.Tensor:
        ctx.cofig = cofig
        m = input.data.shape[0]
        n = input.data.shape[1]
        k = weight.data.shape[0]
        bin_input = torch.LongTensor()
        bin_weight = torch.LongTensor()
        out_tensor = torch.FloatTensor()
        output = Variable(out_tensor, requires_grad = True)
        start_time = time.time()
        binop.encode_rows_cpu(input.data, bin_input)
        binop.encode_rows_cpu(weight.data, bin_weight)
        #print(bin_input.shape, bin_weight.shape)
        
        middle_time = time.time()
        binop.binary_gemm_cpu(bin_input, bin_weight, output.data, m, n, k, 1, 0, 0, 1)
        #print(output.shape)
        
        end_time = time.time()
        ctx.save_for_backward(input, weight)
        #print(bin_input.shape, bin_weight.shape, output.shape)
        print('encode_time:', middle_time - start_time, 'compute_time:', end_time - middle_time)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        cofig = ctx.cofig
        input, weight = ctx.saved_tensors
        bin_grad_1 = torch.LongTensor()
        bin_grad_2 = torch.LongTensor()
        bin_grad_3 = torch.LongTensor()
        bin_grad_4 = torch.LongTensor()
        bin_input = torch.LongTensor()
        grad_a_tensor1 = torch.FloatTensor()
        grad_a_tensor2 = torch.FloatTensor()
        grad_a_tensor3 = torch.FloatTensor()
        grad_a_tensor4 = torch.FloatTensor()
        grad_w_tensor1 = torch.FloatTensor()
        grad_w_tensor2 = torch.FloatTensor()
        grad_w_tensor3 = torch.FloatTensor()
        grad_w_tensor4 = torch.FloatTensor()
        bin_weight = torch.LongTensor()
        bin_grad1 = torch.LongTensor()
        bin_grad2 = torch.LongTensor()
        bin_grad3 = torch.LongTensor()
        bin_grad4 = torch.LongTensor()

        grad_input = grad_output.clone()
        start_time = time.time()
        mask1 = _mask(grad_input)
        mask2 = _mask(grad_input, transpose= False)
        time1 = time.time()
        #print(grad_input)
        weight = weight[mask2]
        input = input[mask1]
        time2 = time.time()
        #print(input.shape)
        grad_a1 = Variable(grad_a_tensor1, requires_grad = False)
        grad_w1 = Variable(grad_w_tensor1, requires_grad = False)
        grad_a2 = Variable(grad_a_tensor2, requires_grad = False)
        grad_w2 = Variable(grad_w_tensor2, requires_grad = False)
        grad_a3 = Variable(grad_a_tensor3, requires_grad = False)
        grad_w3 = Variable(grad_w_tensor3, requires_grad = False)
        grad_a4 = Variable(grad_a_tensor4, requires_grad = False)
        grad_w4 = Variable(grad_w_tensor4, requires_grad = False)

        
        grad_input_weight, zero_point1, scale1= UniformQuantize().apply(grad_input[mask1], cofig.weight_gradient_preconditioner(), True, False)
        grad_input_activation, zero_point2, scale2 = UniformQuantize().apply(grad_input[:,mask2], cofig.activation_gradient_preconditioner(), True, False)
        grad_input_weight = grad_input_weight.to(torch.int)
        grad_input_activation = grad_input_activation.to(torch.int)
        time3 = time.time()
        binop.encode_cols_cpu(weight, bin_weight)
        binop.encode_cols_cpu(input, bin_input)
        
        
        binop.encode_rows_cpu_4(grad_input_activation, bin_grad_1, bin_grad_2, bin_grad_3, bin_grad_4)
        #print(grad_input_weight.shape)
        binop.encode_rows_cpu_4(grad_input_weight, bin_grad1, bin_grad2, bin_grad3, bin_grad4)
        #print(grad_input_weight.shape, bin_grad1.shape)
        
        #print(bin_grad2.shape)
        time4 = time.time()
        #binop.binary_gemm_cpu(bin_grad_1, bin_weight.t(), grad_a1, grad_input.shape[0], grad_input_activation.shape[1], weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_2, bin_weight.t(), grad_a2, grad_input.shape[0], grad_input_activation.shape[1], weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_3, bin_weight.t(), grad_a3, grad_input.shape[0], grad_input_activation.shape[1], weight.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad_4, bin_weight.t(), grad_a4, grad_input.shape[0], grad_input_activation.shape[1], weight.shape[1], 1, 0, 0, alpha)
        binop.binary_gemm_cpu_4(bin_grad_1, bin_grad_2, bin_grad_3, bin_grad_4, bin_weight.t(), grad_a1, grad_input.shape[0], grad_input_activation.shape[1], weight.shape[1], 1, 0, 0)
        #print(bin_grad.shape, bin_weight.shape)
        
        
        #binop.binary_gemm_cpu(bin_grad1, bin_input.t(), grad_w1, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad2, bin_input.t(), grad_w2, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad3, bin_input.t(), grad_w3, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        #binop.binary_gemm_cpu(bin_grad4, bin_input.t(), grad_w4, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0, alpha)
        binop.binary_gemm_cpu_4(bin_grad1, bin_grad2, bin_grad3, bin_grad4, bin_input.t(), grad_w1, bin_grad2.shape[0], grad_input_weight.shape[1],bin_input.shape[1], 1, 0, 0)
        #print(bin_grad1.shape, bin_input.t().shape)
        time5 = time.time()
        bin_weight_sum = weight.sum(0).unsqueeze(0)
        bin_weight_sum = torch.mm(zero_point2, bin_weight_sum)
    
   
        #grad_a1.data = grad_a1.data + grad_a2.data * 2
        #grad_a1.data = grad_a1.data + grad_a3.data * 4 
        #grad_a1.data = grad_a1.data + grad_a4.data * 8
        grad_a1.data = grad_a1.data * scale2.data
        grad_a1.data = grad_a1.data + bin_weight_sum.data       
        bin_input_sum = input.sum(0).unsqueeze(0)
        bin_input_sum = torch.mm(zero_point1, bin_input_sum)        

        #grad_w1.data = grad_w1.data + grad_w2.data * 2
        #grad_w1.data = grad_w1.data + grad_w3.data * 4
        #grad_w1.data = grad_w1.data + grad_w4.data * 8
        grad_w1.data = grad_w1.data * scale1.data
        grad_w1.data = grad_w1.data + bin_input_sum.data 
        time6 = time.time()      
        print('lss_time:', time2 - start_time, 'quantize_time:', time3 - time2, 'encode_time:', time4 - time3, 'mm_time:', time5 - time4, 'dequan_time:', time6 - time5)
        #exit(0)
        return grad_a1, grad_w1, None, None

class BinActive(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, istrain=False, drop=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        #print(in_channels, out_channels)
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1, 1), requires_grad=True)
        self.istrain = istrain
        #self.bn = nn.BatchNorm2d(in_channels)
        self.dropout_ratio = drop

        if drop != 0:
            self.drop = nn.Dropout(drop)


    def forward(self, input):
        #input = self.bn(input)
        if self.istrain:
            input = BinActive.apply(input)
            if self.dropout_ratio != 0:
                input = self.drop(input)
            input = F.conv2d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        else:
            input = bin_conv2d(input, self.weight, self.bias, self.alpha, self.kernel_size, self.stride, self.padding)
            
        return input



class BinLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, istrain=True, drop=0):
        super().__init__(in_features, out_features, bias)

        self.alpha = nn.Parameter(torch.FloatTensor(out_features, 1), requires_grad=False)
        self.istrain = istrain
        self.bn = nn.BatchNorm1d(in_features)
        self.dropout_ratio = drop
        if drop != 0:
            self.drop = nn.Dropout(drop)
        if not istrain:
            #self.weight = nn.Parameter(torch.LongTensor(out_features, 1 + (in_features - 1) // 32))
            print('1')
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.alpha)
        nn.init.normal_(self.bias, mean=0, std=0.01)
    def forward(self, input):
        input = self.bn(input)
        if self.istrain:
            input = BinActive.apply(input)
            if self.dropout_ratio != 0:
                input = self.drop(input)
            input = F.linear(input, weight=self.weight, bias=self.bias)
        else:
            input = bin_linear(input, weight=self.weight, bias=self.bias, alpha=self.alpha)
        return input


class binop_train:
    def __init__(self, model):
        self.alpha_to_save = []
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if type(m).__name__ in ['BinConv2d', 'BinLinear']:
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)
                self.alpha_to_save.append(m.alpha)
        self.num_of_params = len(self.target_modules)

    def binarization(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()

            # meancenter
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(
                self.target_modules[index].data)
            self.target_modules[index].data.add_(negMean)
            # clamp
            self.target_modules[index].data.clamp_(-1.0, 1.0)
            # save param
            self.saved_params[index].copy_(self.target_modules[index].data)

            # get alpha, binarize weight and mutiply alpha
            if len(s) == 4:
                self.alpha_to_save[index].data = \
                    self.target_modules[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1,
                                                                                                      keepdim=True).div(
                        n)
            elif len(s) == 2:
                self.alpha_to_save[index].data = \
                    self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data.sign().mul(
                self.alpha_to_save[index].data.expand(s), out=self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            alpha = self.alpha_to_save[index].data.clone()
            n = weight[0].nelement()
            s = weight.size()
            alpha = alpha.expand(s)
            alpha[weight.le(-1.0)] = 0
            alpha[weight.ge(1.0)] = 0
            alpha = alpha.mul(self.target_modules[index].grad.data)
            add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                add = add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                add = add.sum(1, keepdim=True).div(n).expand(s)
            add = add.mul(weight.sign())
            self.target_modules[index].grad.data = alpha.add(add).mul(1.0 - 1.0 / s[1]).mul(n)

          