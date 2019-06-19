import torch
import torch.nn
import numpy as np



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.mean(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        #loss = y * torch.sqrt(dist_sq)  + (1 - y) * dist
        #loss = torch.sum(loss) / 2.0 / x0.size()[0]
        loss = torch.mean(0.5*loss)
        return loss


class GaussianOverlap(torch.nn.Module):
    """
    log loss based on overlap of two gaussians.

    """

    def __init__(self):
        super(GaussianOverlap, self).__init__()

    def forward(self, diffsquares, y, weights=None, do_batch_mean=True):
        if weights is None:
            weights = torch.ones_like(y)

        ln_area = torch.log(torch.clamp(1.0-torch.abs(torch.erf(torch.sqrt(torch.clamp(diffsquares, 0.0, None)) / 2.0)), 1e-8, 1.0))
        ln_1_min_area = torch.log(torch.clamp(torch.abs(torch.erf(torch.sqrt(torch.clamp(diffsquares, 0.0, None)) / 2.0)), 1e-8, 1.0))

        loss = -y*ln_area - (1-y)*ln_1_min_area
        if not do_batch_mean:
            return loss

        else:
            #print('loss shape, ', loss.size())
            loss = torch.sum(loss*weights)/torch.sum(weights)
            return loss

class AlwaysRight(torch.nn.Module):
    """
    the algorithm is always right loss
    log loss based on overlap of two gaussians.

    """

    def __init__(self):
        super(AlwaysRight, self).__init__()

    def forward(self, mu, y, weights=None, do_batch_mean=True):
        #if all the elements of a row of y are zero, then the algorithm will pick the two vectors
        #which are closest together as the 'true' target pair

        #diff = 0.0*mu[:, :, 0]
        #diff[:, 0] = torch.sqrt(torch.mean(torch.clamp(torch.pow(mu[:, 1] - mu[:, 2], 2), 0.0, None), -1))
        #diff[:, 1] = torch.sqrt(torch.mean(torch.clamp(torch.pow(mu[:, 0] - mu[:, 2], 2), 0.0, None), -1))
        #diff[:, 2] = torch.sqrt(torch.mean(torch.clamp(torch.pow(mu[:, 0]- mu[:, 1], 2), 0.0, None), -1))

        q12 = torch.sqrt(torch.clamp(torch.mean(torch.pow(mu[:, 1] - mu[:, 2], 2), -1), 0.0, None)).unsqueeze(1)
        q02 = torch.sqrt(torch.clamp(torch.mean(torch.pow(mu[:, 0] - mu[:, 2], 2), -1), 0.0, None)).unsqueeze(1)
        q01 = torch.sqrt(torch.clamp(torch.mean(torch.pow(mu[:, 0] - mu[:, 1], 2), -1), 0.0, None)).unsqueeze(1)
        diff = torch.cat((q12, q02, q01), dim=1)

        y_pred = 0.0*mu[:, :, 0]
        y_pred[np.arange(mu.size(0)), torch.min(diff, dim=1)[1]] = 1

        t = torch.sum(y, 1).view(-1, 1).repeat(1, 3)
        y_pred = y_pred*(1-t) + y*t


        ln_area = torch.log(torch.clamp(1.0-torch.abs(torch.erf(diff / 2.0)), 1e-8, 1.0))
        ln_1_min_area = torch.log(torch.clamp(torch.abs(torch.erf(diff / 2.0)), 1e-8, 1.0))

        loss = -y_pred*ln_area - 0.5*(1-y_pred)*ln_1_min_area
        loss = torch.sum(loss, 1)

        if not do_batch_mean:
            return loss

        else:
            #loss = torch.sum(loss*weights)/torch.sum(weights)
            return torch.mean(loss)

