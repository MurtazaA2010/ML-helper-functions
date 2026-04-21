def xtent_loss(z1, z2 , temp):
    batch_size = z1.shape[0]
    z1= F.normalize(z1, dim = 1)
    z2= F.normalize(z2, dim = 1)
    rep = torch.cat([z1, z2],dim  =0)
    sim = torch.matmul(rep, rep.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
    logits = similarity_matrix[~mask].view(2 * batch_size, -1)
    logits /= self.temperature
    labels = torch.arange(batch_size).to(z_i.device)
    labels = torch.cat([labels + batch_size - 1, labels], dim=0)
    labels = torch.arange(2 * batch_size).to(z_i.device)
    labels[::2] += 1
    labels[1::2] -= 1
    target = torch.arange(batch_size).to(z_i.device)
    target = torch.cat([target + batch_size - 1, target], dim=0)

    return self.criterion(logits, target)
