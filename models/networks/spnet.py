import torch
import torch.nn as nn
try:
    from models.networks.architecture import ResnetBlock
except:
    from architecture import ResnetBlock
class SPNet(nn.Module):
    def __init__(self, opt, block=ResnetBlock):
        super().__init__()
        self.n_class = opt.label_nc
        self.resnet_initial_kernel_size = 7
        self.resnet_n_blocks = 9
        ngf = 64
        activation = nn.ReLU(False)

        self.down = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            nn.Conv2d(self.n_class+3, ngf, kernel_size=self.resnet_initial_kernel_size, stride=2, padding=0),
            nn.BatchNorm2d(ngf),
            activation,

            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            activation,

            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            activation,

            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            activation,
        )

        # resnet blocks
        resnet_blocks = []
        for i in range(self.resnet_n_blocks):
            resnet_blocks += [block(ngf*8, norm_layer=nn.BatchNorm2d, kernel_size=3)]
        self.bottle_neck = nn.Sequential(*resnet_blocks)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*8),
            activation,

            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*4),
            activation,

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*2),
            activation,

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            activation,
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            nn.Conv2d(ngf, self.n_class, kernel_size=7, padding=0),
            nn.Softmax2d()
        )
        
    def forward(self, x):
        x = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x)
        out = self.out(x)
        
        return out # shape:

    def generate_fake(self, x):
        return self(x)


if __name__ == '__main__':
    class Opt():
        def __init__(self, label_nc=35):
            self.label_nc = label_nc

    label_nc = 35
    nc = 3
    opt = Opt(label_nc=label_nc)
    x = torch.zeros(2, label_nc+nc, 256, 256).cuda()
    model = SPNet(opt)
    model.cuda()

    out = model(x)

