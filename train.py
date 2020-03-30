import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms

from utils.loadData import ISBITrainDataset, ISBIEvalDataset
from model import Cascade_AttenUnet
import os
import argparse
import scipy.misc
import imageio
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="segmentation")
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lr_sgd', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--output_dir', type=str, default='./results')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--pretrained_idx', type=int, default=300)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--experiment_idx', type=int, default=0)
parser.add_argument('--record', type=int, default=50)

parser.add_argument('--gamma', type=int, default=3)
parser.add_argument('--cv_idx', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()



torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.experiment_idx))
out_dir = os.path.join(args.output_dir, str(args.experiment_idx))
if os.path.exists(checkpoint_dir) is not True:
    os.makedirs(checkpoint_dir)
if os.path.exists(out_dir) is not True:
    os.makedirs(out_dir)
evaluate_dir = os.path.join(out_dir, "evaluate")

evaluate_dir_stage1 = os.path.join(out_dir, "evaluate_stage1")
evaluate_dir_stage2 = os.path.join(out_dir, "evaluate_stage2")


def evaluate(model, dataloader, loss_fn, device, epoch, writer):
    if epoch % args.record == 0:
        save_path = os.path.join(evaluate_dir, "epoch%d" % epoch)
        if os.path.exists(save_path) is not True:
            os.makedirs(save_path)

        save_path_stage1 = os.path.join(evaluate_dir_stage1, "epoch%d" % epoch)
        if os.path.exists(save_path_stage1) is not True:
            os.makedirs(save_path_stage1)

        save_path_stage2 = os.path.join(evaluate_dir_stage2, "epoch%d" % epoch)
        if os.path.exists(save_path_stage2) is not True:
            os.makedirs(save_path_stage2)

    loss_eval = 0.0
    for idx, (eval_img, eval_label) in enumerate(dataloader):
        eval_img = eval_img.float().to(device)
        eval_label = eval_label.float().to(device)
        # pred_label, pred_stage1, pred_stage2, pred_stage3 = model(eval_img)
        pred_label, pred_stage1, pred_stage2 = model(eval_img)
        loss = loss_fn(pred_label, eval_label)
        loss_eval += loss.detach()

        if epoch % args.record ==0:
            pred_label = (pred_label + 1) / 2
            eval_label = (eval_label + 1) / 2
            for i in range(pred_label.size()[0]):
                imageio.imsave(os.path.join(save_path, "%d-pred.png"%i), pred_label[i][0].detach().cpu())
                imageio.imsave(os.path.join(save_path, "%d-tgt.png"%i), eval_label[i][0].detach().cpu())


            pred_stage1 = (pred_stage1 + 1) / 2
            pred_stage2 = (pred_stage2 + 1) / 2
            for i in range(pred_stage1.size()[0]):
                imageio.imsave(os.path.join(save_path_stage1, "%d-pred.png"%i), pred_stage1[i][0].detach().cpu())
                imageio.imsave(os.path.join(save_path_stage1, "%d-tgt.png"%i), eval_label[i][0].detach().cpu())
            for i in range(pred_stage2.size()[0]):
                imageio.imsave(os.path.join(save_path_stage2, "%d-pred.png"%i), pred_stage2[i][0].detach().cpu())
                imageio.imsave(os.path.join(save_path_stage2, "%d-tgt.png"%i), eval_label[i][0].detach().cpu())
        if epoch % 100 == 0:
            for i in range(pred_label.size()[0]):
                writer.add_image('%d-%d'%(epoch, i), pred_label[i])

    return loss_eval / len(dataloader)


def weight_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.constant_(m.bias.data, 0.0)

    elif isinstance(m, nn.ParameterList):
        for var in m:
            nn.init.normal_(var.data, 0.0, 0.02)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

    else:
        pass


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ISBITrainDataset('/home/guoyuze/ICME2019/Dataset/ISBI2012/', transform=transform)
    
    if args.evaluate:
        eval_dataset = ISBIEvalDataset('/home/guoyuze/ICME2019/Dataset/ISBI2012/', transform=transform)

    model = Cascade_AttenUnet(init_depth=12, gamma=args.gamma)
    print("parameters:", sum(param.numel() for param in model.parameters()))
    model.to(device)

    begin_save_idx = 0

    if args.pretrained:
        print("loading pretrained model")
        pretrained_path = os.path.join(checkpoint_dir, "epoch-%d.pth" % args.pretrained_idx)
        model.load_state_dict(torch.load(pretrained_path))
        begin_save_idx = args.pretrained_idx + 1
    else:
        model.apply(weight_init)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 1000], gamma=0.1)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr_sgd, momentum=0.90, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)
    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter()

    for epoch in range(args.epoch):
        scheduler.step()
        dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, drop_last=True)
        epoch_loss = 0.0

        for idx, (train_img, train_label) in enumerate(dataloader):
            optimizer.zero_grad()
            train_img = train_img.float().to(device)
            train_label = train_label.float().to(device)
            pred_label, stage1_pred, stage2_pred = model(train_img)
            loss_final = criterion(pred_label, train_label)
            writer.add_scalar('train_loss', loss_final, len(dataloader) * epoch + idx)
            loss = loss_final
            loss.backward()
            optimizer.step()
            epoch_loss += loss_final.detach()

            if idx % 1 == 0:
                print("epoch %d, iter %d, training loss %.6f"
                      % (epoch + begin_save_idx, idx, loss_final.item()))

        print("=" * 40)
        print("epoch %d, training loss %.6f" %(epoch+begin_save_idx, epoch_loss/len(dataloader)))
        print("=" * 40)

        if args.evaluate:
            eval_dataloader = data.DataLoader(eval_dataset, batch_size=5, shuffle=False)
            eval_loss = evaluate(model, eval_dataloader, criterion, device, epoch, writer)
            writer.add_scalar('eval_loss', eval_loss, epoch)
            print("epoch %d, eval loss %.6f" % (epoch+begin_save_idx, eval_loss))
            print("=" * 40)
            print("=" * 40)
        if epoch % args.record == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch-%d.pth' % (epoch + begin_save_idx)))
    writer.close()

if __name__ == '__main__':
    main()
