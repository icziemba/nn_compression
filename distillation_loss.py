import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.0, temp=1.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temp = temp

    def forward(self, student, teacher, label):
        distillation_loss_func = nn.KLDivLoss()
        student_loss_func = nn.CrossEntropyLoss()
    
        distillation_loss = distillation_loss_func(F.log_softmax(student/self.temp, dim=1),
                                                   F.softmax(teacher/self.temp, dim=1)) * self.alpha
        student_loss = student_loss_func(student, label) * (1.0 - self.alpha)
    
        return distillation_loss + student_loss

