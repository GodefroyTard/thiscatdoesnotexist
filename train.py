import torch
from tqdm import tqdm 
from dataloader import *
from util.util import *
from model import *
from torch.utils.data import DataLoader
import warnings
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    torch.cuda.is_available()
    warnings.filterwarnings("ignore")
    device = 'cuda:0'
    nepoch = 10 
    z_dim = 120
    batch_size = 8
    n_class = 1000
    dataset = CatDataset(['CAT_00','CAT_01','CAT_02','CAT_03','CAT_04','CAT_05','CAT_06'],'data')
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=12,drop_last=True)
    g_lr = 1e-4
    d_lr = 1e-4
    beta1 = 0.0
    beta2 = 0.9
    chn = 16

    G = Generator(chn=chn).to(device)
    D = Discriminator(chn=chn).to(device)

    writer = SummaryWriter()

    g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), d_lr, [beta1, beta2])

    total_iters =0

    for epoch in range(nepoch):
        counter = 0
        for samples in tqdm(dataloader):
            counter +=1
            total_iters += batch_size
            D.train()
            G.train()

            #DISCRIMINATOR =====================
            #fake images
            z = torch.randn(batch_size, z_dim).to(device)
            fake_labels, fake_labels_OH = label_sampel(batch_size,n_class,device)
            fake_images = G(z, fake_labels_OH)
            
            #real images
            real_labels, _ = label_sampel(batch_size,n_class,device)
            real_images = samples['image'].to(device)

            #forward
            d_real = D(real_images,real_labels)
            d_fake = D(fake_images,fake_labels)

            d_loss_real,d_loss_fake = D_hinge(d_real,d_fake)
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            #GENERATOR =======================
            z = torch.randn(batch_size, z_dim).to(device)
            z_class, z_class_one_hot = label_sampel(batch_size,n_class,device)
            
            fake_images = G(z, z_class_one_hot)
            g_out_fake = D(fake_images, z_class)

            g_loss_fake = G_hinge(g_out_fake)

            g_loss_fake.backward()
            g_optimizer.step()
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            #save scalars
            if counter%20 == 0 :
                writer.add_scalar('Loss/D_real', d_loss_real, total_iters)
                writer.add_scalar('Loss/D_fake', d_loss_fake, total_iters)
                writer.add_scalar('Loss/D_total', d_loss, total_iters)
                writer.add_scalar('Loss/G', g_loss_fake, total_iters)

            if counter%100 == 0:
                writer.add_image('fake',makegrid(fake_images),total_iters)
                writer.add_image('real',makegrid(real_images),total_iters)



        save_tensor_batch(fake_images,'img/','fake_epoch'+str(epoch)+'.png' )
        save_tensor_batch(real_images,'img/','real_epoch'+str(epoch)+'.png' )
        print('end of epoch ' + str(epoch) +' over ' + str(nepoch) )
            

    



    print("done")
