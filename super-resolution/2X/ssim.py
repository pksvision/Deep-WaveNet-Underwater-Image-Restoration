from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# from filters import get_gaussian_kernel2d, filter2D

def gaussian(window_size, sigma):
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    x = x.type(torch.FloatTensor)
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size,
                          sigma,
                          force_even: bool = False):
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

def _compute_padding(kernel_size):
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding

def filter2D(input, kernel,
             border_type = 'reflect',
             normalized: bool = False):
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2D(input, kernel)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input_pad = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, h, w)

def ssim(img1, img2, window_size,
         max_val = 1.0, eps = 1e-12):
    r"""Function that computes the Structural Similarity (SSIM) index map between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        img1 (torch.Tensor): the first input image with shape :math:`(B, C, H, W)`.
        img2 (torch.Tensor): the second input image with shape :math:`(B, C, H, W)`.
        window_size (int): the size of the gaussian kernel to smooth the images.
        max_val (float): the dynamic range of the images. Default: 1.
        eps (float): Small value for numerically stability when dividing. Default: 1e-12.

    Returns:
        torch.Tensor: The ssim index map with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim_map = ssim(input1, input2, 5)  # 1x4x5x5
    """
    if not isinstance(img1, torch.Tensor):
        raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                        .format(type(img1)))

    if not isinstance(img2, torch.Tensor):
        raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                        .format(type(img2)))

    if not isinstance(max_val, float):
        raise TypeError("Input max_val type is not a float. Got {}".format(type(max_val)))

    if not len(img1.shape) == 4:
        raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                         .format(img1.shape))

    if not len(img2.shape) == 4:
        raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                         .format(img2.shape))

    if not img1.shape == img2.shape:
        raise ValueError("img1 and img2 shapes must be the same. Got: {} and {}"
                         .format(img1.shape, img2.shape))

    # prepare kernel
    kernel = (
        get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5)).unsqueeze(0)
    )

    # compute coefficients
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1 = filter2D(img1, kernel)
    mu2 = filter2D(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2D(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = filter2D(img2 ** 2, kernel) - mu2_sq
    sigma12 = filter2D(img1 * img2, kernel) - mu1_mu2

    # compute the similarity index map
    num = (2. * mu1_mu2 + C1) * (2. * sigma12 + C2)
    den = (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return num / (den + eps)


def ssim_loss(img1, img2, window_size,
              max_val = 1.0, eps = 1e-12, reduction = 'mean'):
    r"""Function that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim` for details about SSIM.

    Args:
        img1 (torch.Tensor): the first input image with shape :math:`(B, C, H, W)`.
        img2 (torch.Tensor): the second input image with shape :math:`(B, C, H, W)`.
        window_size (int): the size of the gaussian kernel to smooth the images.
        max_val (float): the dynamic range of the images. Default: 1.
        eps (float): Small value for numerically stability when dividing. Default: 1e-12.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'mean'.

    Returns:
        torch.Tensor: The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> loss = ssim_loss(input1, input2, 5)
    """
    # compute the ssim map
    ssim_map = ssim(img1, img2, window_size, max_val, eps)

    # compute and reduce the loss
    loss = torch.clamp((1. - ssim_map) / 2, min=0, max=1)
    # loss = torch.clamp(ssim_map, min=0, max=1)

    # loss = loss.pow_(-1)
    # loss = torch.log10(loss)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    return loss


class SSIM(nn.Module):
    r"""Creates a module that computes the Structural Similarity (SSIM) index between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        window_size (int): the size of the gaussian kernel to smooth the images.
        max_val (float): the dynamic range of the images. Default: 1.
        eps (float): Small value for numerically stability when dividing. Default: 1e-12.

    Shape:
        - Input: :math:`(B, C, H, W)`.
        - Target :math:`(B, C, H, W)`.
        - Output: :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = SSIM(5)
        >>> ssim_map = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(self, window_size, max_val = 1.0, eps = 1e-12) -> None:
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.eps = eps

    def forward(self, img1, img2):
        return ssim(img1, img2, self.window_size, self.max_val, self.eps)


class SSIMLoss(nn.Module):
    r"""Creates a criterion that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim_loss` for details about SSIM.

    Args:
        window_size (int): the size of the gaussian kernel to smooth the images.
        max_val (float): the dynamic range of the images. Default: 1.
        eps (float): Small value for numerically stability when dividing. Default: 1e-12.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'mean'.

    Returns:
        torch.Tensor: The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> criterion = SSIMLoss(5)
        >>> loss = criterion(input1, input2)
    """

    def __init__(self, window_size, max_val = 1.0,
                 eps = 1e-12, reduction = 'mean') -> None:
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.eps = eps
        self.reduction = reduction

    def forward(self, img1, img2):
        return ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction)
