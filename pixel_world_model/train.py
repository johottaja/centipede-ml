"""Independent VAE and dynamics trainers with resumable epoch/update progress."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pixel_world_model.data import TransitionDataset
from pixel_world_model.nets import VisualVAE, ActionLatentDynamics
from pixel_world_model.runtime import (VERSION, Cancellation, atomic_save, load_checkpoint,
                                       fingerprint, device_for, emit, progress)


def load_vae(path, device='cpu'):
    ck = load_checkpoint(path, 'vae', device)
    model = VisualVAE(**ck['architecture']).to(device)
    model.load_state_dict(ck['model'])
    model.eval().requires_grad_(False)
    return model, ck


def compatible(ck, data):
    if ck['dataset_id'] != data.meta['id'] or ck['preprocessing'] != preprocessing(data):
        raise ValueError('Checkpoint does not match the selected dataset/preprocessing. Select the original dataset or start a new run.')


def preprocessing(data):
    return {k: data.meta[k] for k in ('resolution', 'frame_stack', 'frame_gap')}


def latent_cache(data, vae, vae_id, device, batch, cancel):
    root = data.root / 'latent_cache' / vae_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'pairs.npy'
    shape = (data.meta['count'], 2, vae.latent_dim)
    if not path.exists():
        temp = root / 'pairs.partial.npy'
        pairs = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=shape)
        start = time.monotonic()
        with torch.no_grad():
            for offset in range(0, shape[0], batch):
                end = min(offset+batch, shape[0])
                raw = [data.raw(i) for i in range(offset, end)]
                for side, item in enumerate((0, 2)):
                    x = torch.as_tensor(np.stack([r[item] for r in raw]), device=device, dtype=torch.float32)/255
                    pairs[offset:end, side] = vae.encode(x)[0].cpu().numpy()
                pairs.flush()
                progress(start, end, shape[0], phase='latent cache')
                if cancel.cancelled:
                    del pairs
                    temp.unlink(missing_ok=True)
                    return None
        del pairs
        temp.replace(path)
    pairs = np.load(path, mmap_mode='r')
    if pairs.shape != shape:
        raise ValueError('Latent cache shape mismatch; remove the dataset latent_cache directory and retry.')
    return pairs


def train(c, stage):
    cancel = Cancellation()
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    device = device_for(c['device'])
    emit('device', device=str(device))
    data = TransitionDataset(c['dataset'])
    train_indices = TransitionDataset(c['dataset'], 'train').indices
    val_indices = TransitionDataset(c['dataset'], 'validation').indices
    if not len(train_indices) or not len(val_indices):
        raise ValueError('Dataset needs nonempty training and validation splits.')
    prefix = 'vae' if stage == 'vae' else 'dynamics'
    batch, epochs, lr = c[prefix+'_batch'], c[prefix+'_epochs'], c[prefix+'_lr']
    vae_id = None
    if stage == 'vae':
        arch = {'latent_dim': c['latent_dim'], 'img_size': data.meta['resolution']}
        model = VisualVAE(**arch).to(device)
        pairs = None
    else:
        vae, vae_ck = load_vae(c['vae_checkpoint'], device)
        compatible(vae_ck, data)
        vae_id = fingerprint(c['vae_checkpoint'])
        arch = {'latent_dim': vae.latent_dim, 'hidden_width': c['hidden_width'], 'blocks': c['blocks']}
        model = ActionLatentDynamics(**arch).to(device)
        pairs = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch, cursor, updates, best, order = 0, 0, 0, float('inf'), None
    rng = np.random.default_rng(c['seed'])
    resume = c[prefix+'_resume']
    if resume:
        ck = load_checkpoint(resume, stage, device)
        compatible(ck, data)
        if ck['epoch'] > epochs or (ck['epoch'] == epochs and ck['cursor'] > 0):
            raise ValueError('Total epoch budget is below the resumed epoch; increase the budget.')
        if ck['architecture'] != arch or ck.get('vae_id') != vae_id:
            raise ValueError('Resume architecture or VAE differs. Restore the original settings/checkpoint.')
        if ck['config'][prefix+'_batch'] != batch or (stage == 'vae' and ck['config']['beta'] != c['beta']):
            raise ValueError('Resume requires the original batch size and VAE beta.')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        epoch, cursor, updates, best, order = (ck[k] for k in ('epoch', 'cursor', 'updates', 'best', 'order'))
        if stage == 'vae' and ck.get('reconstruction_loss') != 'bce_logits_v1':
            best = float('inf')
            emit('log', text='Switched reconstruction loss to BCE with logits; retaining model/optimizer progress and resetting best validation loss.')
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if device.type == 'mps' and ck.get('mps_rng') is not None:
            torch.mps.set_rng_state(ck['mps_rng'].cpu())
    output = Path(c[prefix+'_output'])
    output.mkdir(parents=True, exist_ok=True)
    # Pin a full copy of the VAE to this dynamics run; paths alone are not sufficient.
    if stage == 'dynamics':
        associated = output / 'associated_vae.pt'
        if associated.exists() and fingerprint(associated) != vae_id:
            raise ValueError('Output directory belongs to a different VAE. Select a new dynamics output directory.')
        if not associated.exists():
            import shutil
            shutil.copyfile(c['vae_checkpoint'], associated)
        pairs = latent_cache(data, vae, vae_id, device, c['vae_batch'], cancel)
        if pairs is None:
            emit('done', cancelled=True, phase='latent cache', checkpoint_saved=False); return

    def checkpoint(name):
        payload = dict(version=VERSION, stage=stage, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       architecture=arch, dataset_id=data.meta['id'], preprocessing=preprocessing(data),
                       config=c, reconstruction_loss='bce_logits_v1' if stage == 'vae' else None, epoch=epoch, cursor=cursor, updates=updates, best=best, order=order,
                       numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), vae_id=vae_id,
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                       mps_rng=torch.mps.get_rng_state() if device.type == 'mps' else None)
        if stage == 'dynamics':
            payload['vae_path'] = 'associated_vae.pt'
        path = output / name
        atomic_save(payload, path)
        emit('checkpoint', path=str(path), epoch=epoch, updates=updates)

    def loss_for(indices, sampling):
        if stage == 'vae':
            x = torch.as_tensor(np.stack([data.raw(int(i))[0] for i in indices]), dtype=torch.float32, device=device)/255
            return model.loss(x, c['beta'], sample=sampling)
        z = torch.tensor(np.asarray(pairs[indices, 0]), device=device)
        target = torch.tensor(np.asarray(pairs[indices, 1]), device=device)
        actions = torch.as_tensor(data.transitions['actions'][indices], device=device)
        loss = F.mse_loss(model(z, actions), target)
        return loss, {'latent_mse': loss.item()}

    start = time.monotonic()
    total = epochs * len(train_indices)
    initial_progress = epoch * len(train_indices) + cursor
    emit('start', stage=stage, total=total)
    while epoch < epochs and not cancel.cancelled:
        if order is None:
            order = rng.permutation(train_indices)
        model.train()
        train_sums = {}
        train_count = 0
        while cursor < len(order) and not cancel.cancelled:
            selected = order[cursor:cursor+batch]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_for(selected, True)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; lower learning rate or inspect the dataset.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            cursor += len(selected); updates += 1
            for key, value in {'loss': loss.item(), **metrics}.items():
                train_sums[key] = train_sums.get(key, 0) + value*len(selected)
            train_count += len(selected)
            if updates % 10 == 0 or cursor == len(order):
                running = {key: value/train_count for key, value in train_sums.items()}
                progress(start, epoch*len(train_indices)+cursor, total, initial=initial_progress, phase=stage, epoch=epoch+1,
                         epoch_position=(epoch*len(train_indices)+cursor)/len(train_indices),
                         updates=updates, **running)
        if cancel.cancelled:
            break
        model.eval()
        sums = {}; count = 0
        validation_start = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(val_indices), batch):
                selected = val_indices[offset:offset+batch]
                loss, metrics = loss_for(selected, False)
                for key, value in {'loss': loss.item(), **metrics}.items():
                    sums[key] = sums.get(key, 0) + value*len(selected)
                count += len(selected)
                if offset % (batch * 10) == 0:
                    progress(validation_start, count, len(val_indices), phase=f'{stage} validation')
                if cancel.cancelled:
                    break
        if cancel.cancelled:
            break
        validation = {k: v/count for k, v in sums.items()}
        if not all(np.isfinite(value) for value in validation.values()):
            raise ValueError('Non-finite validation loss; lower learning rate and inspect the dataset.')
        epoch += 1; cursor = 0; order = None
        emit('validation', epoch=epoch, **validation)
        if validation['loss'] < best:
            best = validation['loss']; checkpoint('best.pt')
        if epoch % c['checkpoint_frequency'] == 0:
            checkpoint(f'epoch_{epoch:04d}.pt')
    checkpoint('cancelled.pt' if cancel.cancelled else 'final.pt')
    emit('done', cancelled=cancel.cancelled, elapsed=time.monotonic()-start, epoch=epoch)


def main():
    from pixel_world_model.cli import main as cli_main
    return cli_main()


if __name__ == '__main__':
    raise SystemExit(main())
