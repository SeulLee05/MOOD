import torch
import numpy as np
import abc
import math
from tqdm import trange

from losses import get_score_fn
from utils.graph_utils import mask_adjs, gen_noise, mask_x
from sde import VPSDE, VESDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    dt = -1. / self.rsde.N

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='adj':
      z = gen_noise(adj, flags, sym=True)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      z = gen_noise(adj, flags, sym=True)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean
    
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    grad = score_fn(x, adj, flags, t)

    if self.obj == 'x':
      for i in range(n_steps):
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps

      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        noise = gen_noise(adj, flags, sym=True)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps

      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not supported")


def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj,
                   weight_x=0, weight_adj=0,
                   predictor='Euler', corrector='Langevin', 
                   snr=0.1, scale_eps=1.0, sampling_steps=1,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', ood=0):

  from models.regressor import RegressorScoreX, RegressorScoreAdj

  def weight_scheduling_fn(weight, t):
    return weight * 0.1 ** t
  
  def total_grad_fn(score_fn, regressor_grad_fn, obj='X'):
    def total_grad(x, adj, flags, t):
      score = score_fn(x, adj, flags, t)

      if obj == 'X':
        weight = weight_x
      else:
        weight = weight_adj

      if weight:
        prop_grad = regressor_grad_fn(x, adj, flags, t)
      else:
        prop_grad = torch.zeros_like(score, device='cuda')
      
      weight_scheduled = weight_scheduling_fn(weight, t[0].item())

      if weight:
        ratio = score.view(x.shape[0], -1).norm(p=1, dim=-1) / prop_grad.view(x.shape[0], -1).norm(p=1, dim=-1)
        weight_scheduled *= ratio[:, None, None]
      
      if isinstance(ood, torch.Tensor):
        score *= (1 - torch.sqrt(ood))
      else:
        score *= (1 - math.sqrt(ood))
      prop_grad *= weight_scheduled
      
      return score + prop_grad
    return total_grad

  def pc_sampler(model_x, model_adj, init_flags, regressor):
    sde_x.change_discreteization_steps(sampling_steps)
    sde_adj.change_discreteization_steps(sampling_steps)

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)
    
    score_fn_x_t = total_grad_fn(score_fn_x, RegressorScoreX(sde_x, regressor), 'X')
    score_fn_adj_t = total_grad_fn(score_fn_adj, RegressorScoreAdj(sde_adj, regressor), 'A')

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x_t, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x_t, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj_t, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj_t, snr, scale_eps, n_steps)

    with torch.no_grad():
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags

      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
      print(' ')

      return (x_mean if denoise else x), (adj_mean if denoise else adj)
  return pc_sampler
