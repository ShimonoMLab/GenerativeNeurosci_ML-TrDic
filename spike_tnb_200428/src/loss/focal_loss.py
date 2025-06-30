import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2., ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input_tensor, target_tensor):
        """
        Args:
            input_tensor: (N=size for each batch, Class, *)
            target_tensor: (N=size for each batch, *)
        """
        kwargs = {
            'ignore_index': self.ignore_index,
        } if self.ignore_index is not None else {}

#        print('-----------------------------------------------------')
#        print('-----------------------------------------------------')
#        print(str(input_tensor))
#        print(input_tensor.shape)
#        print('-----------------------------------------------------')
#        print(kwargs)
#        print('-----------------------------------------------------')
#        print(str(torch.log(input_tensor)))
#        print('-----------------------------------------------------')
#        print('-----------------------------------------------------')

        return F.nll_loss(
            ((1 - input_tensor) ** self.gamma) * torch.log(input_tensor),
            target_tensor,
            **kwargs,
        )


class BinaryFocalLoss(FocalLoss):

    def forward(self, input_tensor, target_tensor):
        """
        Args:
            input_tensor: (N, *), probabilities, not logarithmic
            target_tensor: (N, *)
        """
        assert torch.all(input_tensor >= 0)
        assert torch.all(input_tensor <= 1)
      #  prob = torch.stack([1 - input_tensor, input_tensor]).transpose(0, 1)
        prob = torch.stack([1 - input_tensor, input_tensor]).transpose(0, 1).contiguous()

      #  print('-----------------------------------------------------')
     #   print('-----------------------------------------------------')
     #   print(str(prob))
  #       for i in range(prob.shape[0]):
   #          for j in range(prob.shape[1]):
  #               for k in range(prob.shape[2]):
   #                  for l in range(prob.shape[3]):
    #                     if prob[i][j][k][l]>= 0.5:
    #                         prob[i][j][k][l] = 1
   #                      else:
    #                         prob[i][j][k][l] = 0
    
#        print(' super().forward(prob, target_tensor)  pre-----------------------------------------------------')
#        print(prob)                        
#        print(prob.shape)
#        print(str(target_tensor0))
#        print(target_tensor0.shape)
 #       print(' super().forward(prob, target_tensor)  post-----------------------------------------------------')
        target_tensor0 = target_tensor.clone()
      #  target_tensor0 = target_tensor > 0
        target_tensor0 = (target_tensor > 0).contiguous().long()
#        target_data0 = target_data.clone()
#        target_data0 = target_data0 > 0
        
        print(' super().forward(prob, target_tensor)  pre-----------------------------------------------------')
        print(prob)                        
        print(prob.shape)
        print(str(target_tensor0))
        print(target_tensor0.shape)
        
        for name, tensor in locals().items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
                print(f"{name} is not contiguous")
            
        return super().forward(prob, target_tensor0.long())
    