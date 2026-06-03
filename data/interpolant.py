from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
from motif_scaffolding import twisting


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _trans_endpoint_step(self, d_t, t, trans_1, trans_t):
        step_fraction = d_t / (1 - t)
        return trans_t + step_fraction * (trans_1 - trans_t)

    def _rot_vector_field_scaling(self, t):
        if self._rots_cfg.sample_schedule == 'linear':
            return 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            return self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')

    def _rots_vector_field(self, t, rotmats_1, rotmats_t):
        rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        return self._rot_vector_field_scaling(t) * rot_vf

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        return so3_utils.apply_rotvec_to_rotmat(
            rotmats_t, self._rots_vector_field(t, rotmats_1, rotmats_t) * d_t)

    def _sample_float(self, name, default):
        value = self._sample_cfg.get(name, default)
        if value is None:
            return None
        return float(value)

    def _limit_corrector(self, euler_delta, corrector_delta):
        max_ratio = self._sample_float('max_corrector_norm_ratio', 0.5)
        if max_ratio is None:
            return corrector_delta
        euler_norm = torch.linalg.norm(euler_delta, dim=-1, keepdim=True)
        corrector_norm = torch.linalg.norm(corrector_delta, dim=-1, keepdim=True)
        max_norm = max_ratio * euler_norm
        scale = torch.clamp(max_norm / torch.clamp(corrector_norm, min=1e-8), max=1.0)
        return corrector_delta * scale

    def _limit_multistep_correction(self, euler_delta, correction_delta):
        max_ratio = self._sample_float('max_multistep_correction_ratio', 0.1)
        if max_ratio is None:
            return correction_delta
        euler_norm = torch.linalg.norm(euler_delta, dim=-1, keepdim=True)
        correction_norm = torch.linalg.norm(correction_delta, dim=-1, keepdim=True)
        max_norm = max_ratio * euler_norm
        scale = torch.clamp(max_norm / torch.clamp(correction_norm, min=1e-8), max=1.0)
        return correction_delta * scale

    def _multistep_time_damping(self, t, t_2):
        return torch.clamp((1 - t_2) / (1 - t), min=0.0, max=1.0)

    def _sample_bool(self, name, default):
        value = self._sample_cfg.get(name, default)
        if isinstance(value, str):
            return value.lower() in {'true', '1', 'yes'}
        return bool(value)

    def _ca_ca_deviation(self, trans_t, rotmats_t):
        ca_pos = all_atom.to_atom37(trans_t, rotmats_t)[..., 1, :]
        ca_ca_dists = torch.linalg.norm(ca_pos[:, 1:] - ca_pos[:, :-1], dim=-1)
        return torch.mean(torch.abs(ca_ca_dists - 3.80209737096), dim=-1)

    def _guard_ab2_geometry(
            self,
            trans_euler,
            rotmats_euler,
            trans_ab2,
            rotmats_ab2,
        ):
        if not self._sample_bool('ab2_geometry_guard', True):
            return trans_ab2, rotmats_ab2
        tolerance = self._sample_float('ab2_guard_tolerance', 0.0)
        euler_dev = self._ca_ca_deviation(trans_euler, rotmats_euler)
        ab2_dev = self._ca_ca_deviation(trans_ab2, rotmats_ab2)
        use_ab2 = (ab2_dev <= euler_dev + tolerance)[:, None]
        trans_t_2 = torch.where(use_ab2, trans_ab2, trans_euler)
        rotmats_t_2 = torch.where(use_ab2[..., None], rotmats_ab2, rotmats_euler)
        return trans_t_2, rotmats_t_2

    def _align_rot_vf_to_current(self, prev_rot_vf, prev_rotmats_t, rotmats_t):
        rel_rot = torch.einsum(
            "...ji,...jk->...ik", rotmats_t, prev_rotmats_t)
        return torch.einsum("...ij,...j->...i", rel_rot, prev_rot_vf)

    def _ab2_step(
            self,
            d_t,
            t,
            t_2,
            trans_t,
            rotmats_t,
            pred_trans_1,
            pred_rotmats_1,
            prev_pred_trans_1=None,
            prev_rot_vf=None,
            prev_rotmats_t=None,
        ):
        trans_vf = self._trans_vector_field(t, pred_trans_1, trans_t)
        rot_vf = self._rots_vector_field(t, pred_rotmats_1, rotmats_t)
        trans_euler_delta = self._trans_endpoint_step(
            d_t, t, pred_trans_1, trans_t) - trans_t
        rot_euler_delta = rot_vf * d_t
        trans_euler = trans_t + trans_euler_delta
        rotmats_euler = so3_utils.apply_rotvec_to_rotmat(rotmats_t, rot_euler_delta)

        # The terminal jump lands at the clean endpoint. Multistep extrapolation
        # here tends to inject stale history into the final model call.
        is_terminal_step = torch.isclose(t_2, torch.ones_like(t_2)).item()
        if (
                is_terminal_step
                or prev_pred_trans_1 is None
                or prev_rot_vf is None
                or prev_rotmats_t is None
        ):
            trans_delta = trans_euler_delta
            rot_delta = rot_euler_delta
        else:
            prev_rot_vf = self._align_rot_vf_to_current(
                prev_rot_vf, prev_rotmats_t, rotmats_t)
            time_damping = self._multistep_time_damping(t, t_2)
            trans_weight = self._sample_float('multistep_trans_weight', 0.0)
            rot_weight = self._sample_float('multistep_rot_weight', 0.5)
            trans_weight = trans_weight * time_damping
            rot_weight = rot_weight * time_damping
            trans_correction = (
                trans_weight
                * 0.5
                * d_t / (1 - t)
                * (pred_trans_1 - prev_pred_trans_1)
            )
            rot_correction = rot_weight * 0.5 * (rot_vf - prev_rot_vf) * d_t
            trans_correction = self._limit_multistep_correction(
                trans_euler_delta, trans_correction)
            rot_correction = self._limit_multistep_correction(
                rot_euler_delta, rot_correction)
            trans_delta = trans_euler_delta + trans_correction
            rot_delta = rot_euler_delta + rot_correction

        trans_ab2 = trans_t + trans_delta
        rotmats_ab2 = so3_utils.apply_rotvec_to_rotmat(rotmats_t, rot_delta)
        trans_t_2, rotmats_t_2 = self._guard_ab2_geometry(
            trans_euler, rotmats_euler, trans_ab2, rotmats_ab2)
        history = {
            'pred_trans_1': pred_trans_1.detach(),
            'rot_vf': rot_vf.detach(),
            'rotmats_t': rotmats_t.detach(),
        }
        return trans_t_2, rotmats_t_2, history

    def _safe_endpoint_time(self, t):
        if torch.isclose(t, torch.ones_like(t)).item():
            return torch.ones_like(t) - self._cfg.min_t
        return t

    def _set_model_state(
            self,
            batch,
            num_batch,
            t,
            trans_t,
            rotmats_t,
            trans_1=None,
            rotmats_1=None,
        ):
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t
        batch['so3_t'] = batch['t']
        batch['r3_t'] = batch['t']

    def _heun_step(
            self,
            d_t,
            t_1,
            t_2,
            trans_t_1,
            rotmats_t_1,
            pred_trans_1,
            pred_rotmats_1,
            batch,
            num_batch,
            model,
            motif_scaffolding=False,
            diffuse_mask=None,
            trans_1=None,
            rotmats_1=None,
        ):
        trans_vf_1 = self._trans_vector_field(t_1, pred_trans_1, trans_t_1)
        rot_vf_1 = self._rots_vector_field(t_1, pred_rotmats_1, rotmats_t_1)

        # The flow-matching translation field scales as 1 / (1 - t). A Heun
        # corrector exactly at t=1, or very near it, over-amplifies model error
        # in low-step sampling. Use the Euler predictor for this terminal jump
        # and let the final model call below project to the clean endpoint.
        if torch.isclose(t_2, torch.ones_like(t_2)).item():
            trans_t_2 = self._trans_endpoint_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = so3_utils.apply_rotvec_to_rotmat(
                rotmats_t_1, rot_vf_1 * d_t)
            return trans_t_2, rotmats_t_2, pred_trans_1, pred_rotmats_1

        trans_t_2_pred = self._trans_endpoint_step(
            d_t, t_1, pred_trans_1, trans_t_1)
        rot_delta_1 = rot_vf_1 * d_t
        rotmats_t_2_pred = so3_utils.apply_rotvec_to_rotmat(
            rotmats_t_1, rot_delta_1)
        if motif_scaffolding:
            trans_t_2_pred = _trans_diffuse_mask(trans_t_2_pred, trans_1, diffuse_mask)
            rotmats_t_2_pred = _rots_diffuse_mask(rotmats_t_2_pred, rotmats_1, diffuse_mask)

        t_2_model = self._safe_endpoint_time(t_2)
        self._set_model_state(
            batch, num_batch, t_2_model, trans_t_2_pred, rotmats_t_2_pred, trans_1, rotmats_1)
        with torch.no_grad():
            model_out_2 = model(batch)
        pred_trans_1_2 = model_out_2['pred_trans']
        pred_rotmats_1_2 = model_out_2['pred_rotmats']

        rot_vf_2 = self._rots_vector_field(t_2_model, pred_rotmats_1_2, rotmats_t_2_pred)

        corrector_weight = self._sample_float('corrector_weight', 1.0)
        trans_euler_delta = trans_t_2_pred - trans_t_1
        trans_corrector_delta = (
            corrector_weight
            * d_t / (1 - t_1)
            * 0.5
            * (pred_trans_1_2 - pred_trans_1)
        )
        trans_corrector_delta = self._limit_corrector(
            trans_euler_delta, trans_corrector_delta)
        trans_t_2 = trans_t_1 + trans_euler_delta + trans_corrector_delta

        rot_corrector_delta = corrector_weight * 0.5 * (rot_vf_2 - rot_vf_1) * d_t
        rot_corrector_delta = self._limit_corrector(rot_delta_1, rot_corrector_delta)
        rotmats_t_2 = so3_utils.apply_rotvec_to_rotmat(
            rotmats_t_1, rot_delta_1 + rot_corrector_delta)
        return trans_t_2, rotmats_t_2, pred_trans_1_2, pred_rotmats_1_2

    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            trans_1=None,
            rotmats_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            verbose=False,
        ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': res_mask,
            'res_idx': res_idx 
        }

        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
        if motif_scaffolding and not self._cfg.twisting.use: # amortisation
            diffuse_mask = diffuse_mask.expand(num_batch, -1) # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError('NaN detected in trans_0')

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use: # sampling / guidance
            assert trans_1.shape[0] == 1 # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = twisting.find_ranges_and_lengths(motif_locations)

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = true_motif_locations if self._cfg.twisting.motif_loc else None
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(num_res, motif_segments_length, motif_locations=motif_locations, num_rots=self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, trans_motif=trans_motif, R_motif=R_motif, max_offsets=self._cfg.twisting.max_offsets, device=self._device, dtype=torch.float64, return_rots=False)

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        sampling_method = self._sample_cfg.get('method', 'euler')
        if sampling_method not in {'euler', 'heun', 'ab2'}:
            raise ValueError(f'Unknown sampling method: {sampling_method}')
        if sampling_method in {'heun', 'ab2'} and trans_potential is not None:
            raise ValueError(f'{sampling_method} sampling does not support trans_potential yet.')
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        ab2_history = None
        for i, t_2 in enumerate(ts[1:]):
            if verbose: # and i % 1 == 0:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(trans_0.device), torch.cuda.memory_allocated(trans_0.device))
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            self._set_model_state(
                batch, num_batch, t_1, trans_t_1, rotmats_t_1, trans_1, rotmats_1)
            d_t = t_2 - t_1

            use_twisting = motif_scaffolding and self._cfg.twisting.use and t_1 >= self._cfg.twisting.t_min
            if sampling_method in {'heun', 'ab2'} and use_twisting:
                raise ValueError(f'{sampling_method} sampling does not support twisting guidance yet.')

            if use_twisting: # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch['r3_t'] #TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj)

            else:
                with torch.no_grad():
                    model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            if self._cfg.self_condition:
                if motif_scaffolding:
                    batch['trans_sc'] = (
                        pred_trans_1 * diffuse_mask[..., None]
                        + trans_1 * (1 - diffuse_mask[..., None])
                    )
                else:
                    batch['trans_sc'] = pred_trans_1

            # Take reverse step
            if sampling_method == 'euler':
                trans_t_2 = self._trans_euler_step(
                    d_t, t_1, pred_trans_1, trans_t_1)
                if trans_potential is not None:
                    with torch.inference_mode(False):
                        grad_pred_trans_1 = pred_trans_1.clone().detach().requires_grad_(True)
                        pred_trans_potential = autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
                    if self._trans_cfg.potential_t_scaling:
                        trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                    else:
                        trans_t_2 -= pred_trans_potential * d_t
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)
                clean_trans_1 = pred_trans_1
                clean_rotmats_1 = pred_rotmats_1
            elif sampling_method == 'heun':
                trans_t_2, rotmats_t_2, clean_trans_1, clean_rotmats_1 = self._heun_step(
                    d_t,
                    t_1,
                    t_2,
                    trans_t_1,
                    rotmats_t_1,
                    pred_trans_1,
                    pred_rotmats_1,
                    batch,
                    num_batch,
                    model,
                    motif_scaffolding=motif_scaffolding,
                    diffuse_mask=diffuse_mask,
                    trans_1=trans_1,
                    rotmats_1=rotmats_1,
                )
                if self._cfg.self_condition:
                    if motif_scaffolding:
                        batch['trans_sc'] = (
                            clean_trans_1 * diffuse_mask[..., None]
                            + trans_1 * (1 - diffuse_mask[..., None])
                        )
                    else:
                        batch['trans_sc'] = clean_trans_1
            elif sampling_method == 'ab2':
                trans_t_2, rotmats_t_2, ab2_history = self._ab2_step(
                    d_t,
                    t_1,
                    t_2,
                    trans_t_1,
                    rotmats_t_1,
                    pred_trans_1,
                    pred_rotmats_1,
                    prev_pred_trans_1=None if ab2_history is None else ab2_history['pred_trans_1'],
                    prev_rot_vf=None if ab2_history is None else ab2_history['rot_vf'],
                    prev_rotmats_t=None if ab2_history is None else ab2_history['rotmats_t'],
                )
                clean_trans_1 = pred_trans_1
                clean_rotmats_1 = pred_rotmats_1
            clean_traj.append(
                (clean_trans_1.detach().cpu(), clean_rotmats_1.detach().cpu())
            )
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        self._set_model_state(
            batch, num_batch, t_1, trans_t_1, rotmats_t_1, trans_1, rotmats_1)
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj

    def guidance(self, trans_t, rotmats_t, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out['pred_trans'][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out['pred_rotmats'][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(trans_motif, R_motif, self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, device=self._device, dtype=torch.float32)

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(R_pred, trans_pred, R_motif, trans_motif, Log_delta_R, delta_x, None, None, None, F, twist_potential_rot=self._cfg.twisting.potential_rot, twist_potential_trans=self._cfg.twisting.potential_trans)

        with torch.no_grad():
            # Choose scaling
            t_trans = t
            t_so3 = t
            if self._cfg.twisting.scale_w_t == 'ot':
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'linear':
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'constant':
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            var_trans = var_trans + self._cfg.twisting.obs_noise ** 2
            var_rot = var_rot + self._cfg.twisting.obs_noise ** 2

            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # Compute update
            trans_t, rotmats_t = twisting.step(trans_t, rotmats_t, grad_x_log_p_motif, grad_Log_delta_R, d_t, trans_scale_t, rot_scale_t, self._cfg.twisting.update_trans, self._cfg.twisting.update_rot)

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj
