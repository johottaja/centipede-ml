"""Interactive held-out reconstruction and autoregressive rollout inspection."""
from pathlib import Path
import numpy as np
import pygame
import torch
from pixel_world_model.data import TransitionDataset
from pixel_world_model.nets import ActionLatentDynamics, reconstruction_loss
from pixel_world_model.train import load_vae, compatible
from pixel_world_model.runtime import load_checkpoint, fingerprint, device_for, emit


def tensor(stack, device):
    return torch.as_tensor(stack, dtype=torch.float32, device=device).unsqueeze(0)/255


@torch.no_grad()
def predict_sequence(vae, dynamics, initial, actions, device):
    """Encode exactly once. Never teacher-force actual future frames."""
    z = vae.encode(tensor(initial, device))[0]
    latents = [z]
    decoded = [vae.decode(z)[0].cpu().numpy()]
    for action in actions:
        z = dynamics(z, torch.tensor([action], device=device))
        latents.append(z)
        decoded.append(vae.decode(z)[0].cpu().numpy())
    return latents, decoded


class Inspection:
    def __init__(self, c, stage):
        self.c, self.stage = c, stage
        self.device = device_for(c['device'])
        self.data = TransitionDataset(c['dataset'], 'validation')
        if not len(self.data):
            raise ValueError('No held-out samples in this dataset.')
        self.dynamics = None
        if stage == 'vae':
            self.vae, ck = load_vae(c['vae_checkpoint'], self.device)
        else:
            path = Path(c['dynamics_checkpoint'])
            ck = load_checkpoint(path, 'dynamics', self.device)
            compatible(ck, self.data)
            vae_path = path.parent / ck['vae_path']
            if not vae_path.exists() or fingerprint(vae_path) != ck['vae_id']:
                raise ValueError('Associated VAE missing or changed. Restore associated_vae.pt from the original dynamics run.')
            self.vae, vae_ck = load_vae(vae_path, self.device)
            compatible(vae_ck, self.data)
            self.dynamics = ActionLatentDynamics(**ck['architecture']).to(self.device)
            self.dynamics.load_state_dict(ck['model']); self.dynamics.eval()
        compatible(ck, self.data)
        self.position, self.step, self.channel = 0, 0, 3
        self.horizon = c['horizon']
        self.playing = False
        self.prepare()

    @torch.no_grad()
    def prepare(self):
        self.index = int(self.data.indices[self.position])
        state, _, _ = self.data.raw(self.index)
        self.state = state.astype(np.float32)/255
        self.step = 0
        if self.stage == 'vae':
            logits = self.vae.forward_logits(tensor(state, self.device), sample=False)[0]
            self.recon_logits = logits[0].cpu().numpy()
            self.recon = logits.sigmoid()[0].cpu().numpy()
        else:
            self.trajectory = self.data.trajectory(self.index, self.horizon)
            self.actions = [t[1] for t in self.trajectory]
            self.latents, self.decoded = predict_sequence(self.vae, self.dynamics, state, self.actions, self.device)

    def advance(self, amount):
        if self.stage == 'vae':
            self.position = (self.position+amount) % len(self.data)
            self.prepare()
        else:
            self.step = max(0, min(len(self.actions), self.step+amount))

    def next_trajectory(self):
        current = int(self.data.transitions['episodes'][self.index])
        candidates = [p for p, i in enumerate(self.data.indices)
                      if int(self.data.transitions['episodes'][i]) != current and p > self.position]
        self.position = candidates[0] if candidates else 0
        self.prepare()

    @torch.no_grad()
    def panels(self):
        channel = self.channel
        if self.stage == 'vae':
            _, metrics = reconstruction_loss(torch.from_numpy(self.recon_logits)[None],
                                             torch.from_numpy(self.state)[None])
            return [('Original', self.state[channel]), ('Reconstruction', self.recon[channel]),
                    ('Absolute difference', np.abs(self.state[channel]-self.recon[channel]))], f"BCE={metrics['reconstruction']:.6f} | pixel MSE={metrics['pixel_mse']:.6f}"
        actual = self.state if self.step == 0 else self.trajectory[self.step-1][2].astype(np.float32)/255
        pred = self.decoded[self.step]
        target = self.vae.encode(torch.as_tensor(actual, device=self.device).unsqueeze(0))[0]
        latent_mse = (target-self.latents[self.step]).square().mean().item()
        mse = float(np.mean((actual-pred)**2))
        return [('Initial', self.state[channel]), ('Initial reconstruction', self.decoded[0][channel]),
                (f'Actual +{self.step}', actual[channel]), (f'Predicted +{self.step}', pred[channel])], \
               f'step={self.step}/{len(self.actions)}  latent MSE={latent_mse:.6f}  pixel MSE={mse:.6f}  actions={self.actions[:self.step]}'


def draw(view, screen, font):
    panels, metrics = view.panels()
    screen.fill((18, 20, 28))
    width, height = screen.get_size()
    tile = min(width//len(panels)-16, height-165)
    for i, (title, frame) in enumerate(panels):
        x = i*(width//len(panels))+8
        screen.blit(font.render(title, True, (230, 235, 245)), (x, 12))
        gray = np.clip(frame*255, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surface, (tile, tile)), (x, 42))
    lines = [metrics, f'held-out sample {view.position+1}/{len(view.data)} | stack frame {view.channel+1}/4 | horizon {view.horizon} | {"playing" if view.playing else "paused"}',
             'Space: play/pause | Left/Right: step | 1–4: stack frame | [ / ]: horizon | N: next trajectory | R: reset',
             'Mouse: click controls below | Esc/Q: quit']
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (200, 208, 222)), (10, tile+52+i*22))
    controls = [('Play/Pause', 'play'), ('Previous', 'previous'), ('Next', 'next'), ('Horizon −', 'less'),
                ('Horizon +', 'more'), ('Frame', 'frame'), ('Trajectory', 'trajectory'), ('Reset', 'reset')]
    buttons = []
    for i, (label, action) in enumerate(controls):
        rect = pygame.Rect(8+i*(width//8), height-36, width//8-12, 28)
        pygame.draw.rect(screen, (48, 57, 76), rect, border_radius=4)
        screen.blit(font.render(label, True, (235, 240, 250)), (rect.x+5, rect.y+5))
        buttons.append((rect, action))
    pygame.display.flip()
    return buttons


def action(view, command):
    if command == 'play': view.playing = not view.playing
    elif command == 'previous': view.advance(-1)
    elif command == 'next': view.advance(1)
    elif command == 'frame': view.channel = (view.channel+1) % 4
    elif command == 'trajectory': view.next_trajectory()
    elif command == 'reset': view.prepare()
    elif command in ('less', 'more'):
        view.horizon = max(1, min(100, view.horizon + (1 if command == 'more' else -1)))
        view.prepare()


def visualize(c, stage):
    view = Inspection(c, stage)
    emit('device', device=str(view.device))
    pygame.init()
    count = 3 if stage == 'vae' else 4
    screen = pygame.display.set_mode((min(1500, count*340), 520), pygame.RESIZABLE)
    pygame.display.set_caption(f'Pixel world model — {stage} inspection')
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock(); last = pygame.time.get_ticks()
    buttons = draw(view, screen, font)
    keys = {pygame.K_SPACE: 'play', pygame.K_LEFT: 'previous', pygame.K_RIGHT: 'next',
            pygame.K_LEFTBRACKET: 'less', pygame.K_RIGHTBRACKET: 'more', pygame.K_n: 'trajectory', pygame.K_r: 'reset'}
    try:
        running = True
        while running:
            changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                    elif pygame.K_1 <= event.key <= pygame.K_4: view.channel = event.key-pygame.K_1
                    elif event.key in keys: action(view, keys[event.key])
                    changed = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for rect, command in buttons:
                        if rect.collidepoint(event.pos): action(view, command); changed = True
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(800, event.w), max(480, event.h)), pygame.RESIZABLE)
                    changed = True
            now = pygame.time.get_ticks()
            if view.playing and now-last >= 200:
                if stage == 'dynamics' and view.step == len(view.actions):
                    view.playing = False
                else:
                    view.advance(1)
                last = now; changed = True
            if changed:
                buttons = draw(view, screen, font)
            clock.tick(30)
    finally:
        pygame.quit()


if __name__ == '__main__':
    from pixel_world_model.cli import main
    raise SystemExit(main())
