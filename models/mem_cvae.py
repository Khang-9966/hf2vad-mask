import torch
from torch import nn
from models.vunet import VUnet
from models.ml_memAE_sc import ML_MemAE_SC
from torch.nn import ModuleDict, ModuleList

class HFVAD(nn.Module):
    """
    ML-MemAE-SC + CVAE
    """

    def __init__(self, num_hist, num_pred, config, features_root, num_slots, shrink_thres, skip_ops, mem_usage,
                 finetune=False):
        super(HFVAD, self).__init__()

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.features_root = features_root
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.skip_ops = skip_ops
        self.mem_usage = mem_usage
        self.finetune = finetune

        self.x_ch = 3  # num of RGB channels
        self.mask_y_ch = 1  # num of optical flow channels
        self.flow_y_ch = 2  # num of optical flow channels

        self.memAE = ModuleDict()
        self.memAE["mask"] = ML_MemAE_SC(num_in_ch=self.mask_y_ch, seq_len=1, features_root=self.features_root,
                                 num_slots=self.num_slots, shrink_thres=self.shrink_thres,
                                 mem_usage=self.mem_usage,
                                 skip_ops=self.skip_ops)

        self.memAE["flow"] = ML_MemAE_SC(num_in_ch=self.flow_y_ch, seq_len=1, features_root=self.features_root,
                                 num_slots=self.num_slots, shrink_thres=self.shrink_thres,
                                 mem_usage=self.mem_usage,
                                 skip_ops=self.skip_ops)

        self.vunet = VUnet(config)

        self.mse_loss = nn.MSELoss()

    def forward(self, sample_frame, sample_mask, sample_of, mode="train"):
        """
        :param sample_frame: 5 frames in a video clip
        :param sample_of: 4 corresponding flows
        :return:
        """

        mask_att_weight3_cache, mask_att_weight2_cache, mask_att_weight1_cache = [], [], []
        mask_recon = torch.zeros_like(sample_mask)
        # reconstruct flows
        for j in range(self.num_hist):
            mask_memAE_out = self.memAE["mask"](sample_frame[:, 3 * j:3 * (j + 1), :, :])
            mask_recon[:, 1 * j:1 * (j + 1), :, :] = mask_memAE_out["recon"]
            mask_att_weight3_cache.append(mask_memAE_out["att_weight3"])
            mask_att_weight2_cache.append(mask_memAE_out["att_weight2"])
            mask_att_weight1_cache.append(mask_memAE_out["att_weight1"])
        mask_att_weight3 = torch.cat(mask_att_weight3_cache, dim=0)
        mask_att_weight2 = torch.cat(mask_att_weight2_cache, dim=0)
        mask_att_weight1 = torch.cat(mask_att_weight1_cache, dim=0)
        if self.finetune:
            mask_loss_recon = self.mse_loss(mask_recon, sample_mask)
            mask_loss_sparsity = torch.mean(
                torch.sum(-mask_att_weight3 * torch.log(mask_att_weight3 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-mask_att_weight2 * torch.log(mask_att_weight2 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-mask_att_weight1 * torch.log(mask_att_weight1 + 1e-12), dim=1)
            )

        flow_att_weight3_cache, flow_att_weight2_cache, flow_att_weight1_cache = [], [], []
        flow_recon = torch.zeros_like(sample_of)
        # reconstruct flows
        for j in range(self.num_hist):
          flow_memAE_out = self.memAE["flow"](sample_frame[:, 3 * j:3 * (j + 2), :, :])
          flow_recon[:, 2 * j:2 * (j + 1), :, :] = flow_memAE_out["recon"]
          flow_att_weight3_cache.append(flow_memAE_out["att_weight3"])
          flow_att_weight2_cache.append(flow_memAE_out["att_weight2"])
          flow_att_weight1_cache.append(flow_memAE_out["att_weight1"])
        flow_att_weight3 = torch.cat(flow_att_weight3_cache, dim=0)
        flow_att_weight2 = torch.cat(flow_att_weight2_cache, dim=0)
        flow_att_weight1 = torch.cat(flow_att_weight1_cache, dim=0)
        if self.finetune:
          flow_loss_recon = self.mse_loss(flow_recon, sample_of)
          flow_loss_sparsity = torch.mean(
              torch.sum(-flow_att_weight3 * torch.log(flow_att_weight3 + 1e-12), dim=1)
          ) + torch.mean(
              torch.sum(-flow_att_weight2 * torch.log(flow_att_weight2 + 1e-12), dim=1)
          ) + torch.mean(
              torch.sum(-flow_att_weight1 * torch.log(flow_att_weight1 + 1e-12), dim=1)
          )

        frame_in = sample_frame[:, :-self.x_ch * self.num_pred, :, :]
        frame_target = sample_frame[:, -self.x_ch * self.num_pred:, :, :]

        input_dict = dict(appearance=frame_in, motion=flow_recon, mask=mask_recon)
        frame_pred = self.vunet(input_dict, mode=mode)

        out = dict(frame_pred=frame_pred, frame_target=frame_target,
                   of_recon=flow_recon, of_target=sample_of, mask_recon=mask_recon, mask_target=sample_mask)
        out.update(self.vunet.saved_tensors)

        if self.finetune:
            ML_MemAE_SC_dict = dict(mask_loss_recon=mask_loss_recon, mask_loss_sparsity=mask_loss_sparsity,
            flow_loss_recon=flow_loss_recon, flow_loss_sparsity=flow_loss_sparsity)
            out.update(ML_MemAE_SC_dict)

        return out
