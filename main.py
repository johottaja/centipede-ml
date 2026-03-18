import sys
import random
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE = 16
COLS, ROWS = 30, 31
PLAYER_ZONE_TOP = ROWS - 5  # player can only move in bottom 5 rows
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE
FPS = 60

COLOR_BG = (0, 0, 0)
COLOR_MUSHROOM = [(34, 139, 34), (0, 100, 0), (0, 60, 0), (60, 40, 20)]
COLOR_PLAYER = (0, 200, 255)
COLOR_BULLET = (255, 255, 100)
COLOR_CENTIPEDE_HEAD = (255, 50, 50)
COLOR_CENTIPEDE_BODY = (200, 0, 0)
COLOR_HUD = (255, 255, 255)

MUSHROOM_HP = 4
CENTIPEDE_LENGTH = 12
CENTIPEDE_SPEED = 2  # pixels per frame
PLAYER_SPEED = 4
BULLET_SPEED = 8
SHOOT_COOLDOWN = 8  # frames between shots


# ---------------------------------------------------------------------------
# Mushroom
# ---------------------------------------------------------------------------
class Mushroom:
    def __init__(self, col, row):
        self.col = col
        self.row = row
        self.hp = MUSHROOM_HP

    @property
    def rect(self):
        return pygame.Rect(self.col * TILE, self.row * TILE, TILE, TILE)

    def draw(self, surf):
        stage = max(0, min(3, MUSHROOM_HP - self.hp))
        color = COLOR_MUSHROOM[stage]
        r = self.rect
        # draw a rounded-ish mushroom shape that shrinks with damage
        shrink = stage * 2
        inner = r.inflate(-shrink, -shrink)
        pygame.draw.ellipse(surf, color, inner)

    def hit(self):
        self.hp -= 1
        return self.hp <= 0


# ---------------------------------------------------------------------------
# Mushroom field  – grid-based for O(1) lookups
# ---------------------------------------------------------------------------
class MushroomField:
    def __init__(self):
        self.grid: dict[tuple[int, int], Mushroom] = {}

    def populate(self, density=0.06):
        for r in range(1, PLAYER_ZONE_TOP):
            for c in range(COLS):
                if random.random() < density:
                    self.grid[(c, r)] = Mushroom(c, r)

    def get(self, col, row):
        return self.grid.get((col, row))

    def remove(self, col, row):
        self.grid.pop((col, row), None)

    def add(self, col, row):
        if 0 <= col < COLS and 0 <= row < ROWS and (col, row) not in self.grid:
            self.grid[(col, row)] = Mushroom(col, row)

    def draw(self, surf):
        for m in self.grid.values():
            m.draw(surf)

    def collides(self, rect):
        c1 = rect.left // TILE
        c2 = rect.right // TILE
        r1 = rect.top // TILE
        r2 = rect.bottom // TILE
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                m = self.grid.get((c, r))
                if m and m.rect.colliderect(rect):
                    return m
        return None


# ---------------------------------------------------------------------------
# Centipede segment
# ---------------------------------------------------------------------------
class Segment:
    def __init__(self, x, y, is_head=False):
        self.x = float(x)
        self.y = float(y)
        self.is_head = is_head
        self.dir = 1  # 1 = right, -1 = left
        self.dropping = 0  # remaining px to drop
        self.speed = CENTIPEDE_SPEED

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), TILE, TILE)

    @property
    def col(self):
        return int(self.x) // TILE

    @property
    def row(self):
        return int(self.y) // TILE

    def draw(self, surf):
        color = COLOR_CENTIPEDE_HEAD if self.is_head else COLOR_CENTIPEDE_BODY
        r = self.rect
        pygame.draw.ellipse(surf, color, r)
        if self.is_head:
            # eyes
            ex = r.centerx + self.dir * 3
            pygame.draw.circle(surf, (255, 255, 255), (ex, r.centery - 2), 2)
            pygame.draw.circle(surf, (255, 255, 255), (ex, r.centery + 2), 2)


# ---------------------------------------------------------------------------
# Centipede chain
# ---------------------------------------------------------------------------
class Centipede:
    """One chain of segments that moves together."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        if segments:
            segments[0].is_head = True

    def update(self, field: MushroomField):
        for seg in self.segments:
            if seg.dropping > 0:
                dy = min(seg.speed, seg.dropping)
                seg.y += dy
                seg.dropping -= dy
                if seg.dropping == 0:
                    seg.dir *= -1
                continue

            seg.x += seg.dir * seg.speed
            next_col = int(seg.x + (TILE if seg.dir == 1 else -1)) // TILE

            hit_wall = next_col < 0 or next_col >= COLS
            hit_mush = (
                not hit_wall and field.get(next_col, seg.row) is not None
            )

            if hit_wall or hit_mush:
                seg.x = seg.col * TILE  # snap
                seg.dropping = TILE

    def draw(self, surf):
        for seg in reversed(self.segments):
            seg.draw(surf)


# ---------------------------------------------------------------------------
# Bullet
# ---------------------------------------------------------------------------
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    @property
    def rect(self):
        return pygame.Rect(self.x - 1, self.y - 4, 3, 8)

    def update(self):
        self.y -= BULLET_SPEED
        if self.y < -8:
            self.alive = False

    def draw(self, surf):
        pygame.draw.rect(surf, COLOR_BULLET, self.rect)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class Player:
    def __init__(self):
        self.x = WIDTH // 2 - TILE // 2
        self.y = (ROWS - 2) * TILE
        self.cooldown = 0
        self.lives = 3

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, TILE, TILE)

    def update(self, keys):
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= PLAYER_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += PLAYER_SPEED
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= PLAYER_SPEED
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += PLAYER_SPEED

        self.x = max(0, min(WIDTH - TILE, self.x + dx))
        self.y = max(PLAYER_ZONE_TOP * TILE, min(HEIGHT - TILE, self.y + dy))
        if self.cooldown > 0:
            self.cooldown -= 1

    def shoot(self):
        if self.cooldown > 0:
            return None
        self.cooldown = SHOOT_COOLDOWN
        return Bullet(self.x + TILE // 2, self.y)

    def draw(self, surf):
        r = self.rect
        # triangular shooter shape
        pts = [
            (r.centerx, r.top),
            (r.left, r.bottom),
            (r.right, r.bottom),
        ]
        pygame.draw.polygon(surf, COLOR_PLAYER, pts)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Centipede")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.reset()

    def reset(self):
        self.score = 0
        self.player = Player()
        self.bullets: list[Bullet] = []
        self.centipedes: list[Centipede] = []
        self.field = MushroomField()
        self.field.populate()
        self.spawn_centipede()
        self.game_over = False

    def spawn_centipede(self):
        segs = []
        start_x = (COLS // 2) * TILE
        for i in range(CENTIPEDE_LENGTH):
            segs.append(Segment(start_x - i * TILE, 0, is_head=(i == 0)))
        self.centipedes.append(Centipede(segs))

    # ---- update helpers ---------------------------------------------------
    def _handle_bullet_mushroom(self):
        for b in self.bullets:
            if not b.alive:
                continue
            m = self.field.collides(b.rect)
            if m:
                b.alive = False
                if m.hit():
                    self.field.remove(m.col, m.row)
                    self.score += 5
                else:
                    self.score += 1

    def _handle_bullet_centipede(self):
        new_centipedes: list[Centipede] = []
        for chain in self.centipedes:
            hit_idx = None
            for b in self.bullets:
                if not b.alive:
                    continue
                for i, seg in enumerate(chain.segments):
                    if b.rect.colliderect(seg.rect):
                        b.alive = False
                        hit_idx = i
                        break
                if hit_idx is not None:
                    break

            if hit_idx is not None:
                seg = chain.segments[hit_idx]
                self.field.add(seg.col, seg.row)
                self.score += 100 if seg.is_head else 10

                before = chain.segments[:hit_idx]
                after = chain.segments[hit_idx + 1:]
                if before:
                    new_centipedes.append(Centipede(before))
                if after:
                    after[0].is_head = True
                    new_centipedes.append(Centipede(after))
            else:
                new_centipedes.append(chain)
        self.centipedes = new_centipedes

    def _handle_player_centipede(self):
        for chain in self.centipedes:
            for seg in chain.segments:
                if self.player.rect.colliderect(seg.rect):
                    self.player.lives -= 1
                    if self.player.lives <= 0:
                        self.game_over = True
                    else:
                        self.player.x = WIDTH // 2 - TILE // 2
                        self.player.y = (ROWS - 2) * TILE
                    return

    def _check_respawn(self):
        if not self.centipedes:
            self.spawn_centipede()

    # ---- main loop --------------------------------------------------------
    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_r and self.game_over:
                        self.reset()

            if not self.game_over:
                keys = pygame.key.get_pressed()
                self.player.update(keys)

                if keys[pygame.K_SPACE] and not self.bullets:
                    b = self.player.shoot()
                    if b:
                        self.bullets.append(b)

                for b in self.bullets:
                    b.update()
                self.bullets = [b for b in self.bullets if b.alive]

                for c in self.centipedes:
                    c.update(self.field)

                self._handle_bullet_mushroom()
                self._handle_bullet_centipede()
                self._handle_player_centipede()
                self._check_respawn()

            # ---- draw ---------------------------------------------------------
            self.screen.fill(COLOR_BG)
            self.field.draw(self.screen)
            for c in self.centipedes:
                c.draw(self.screen)
            for b in self.bullets:
                b.draw(self.screen)
            if not self.game_over:
                self.player.draw(self.screen)

            # HUD
            hud = self.font.render(
                f"Score: {self.score}   Lives: {self.player.lives}", True, COLOR_HUD
            )
            self.screen.blit(hud, (8, HEIGHT - 20))

            if self.game_over:
                go_text = self.font.render("GAME OVER – press R to restart", True, COLOR_HUD)
                self.screen.blit(
                    go_text,
                    (WIDTH // 2 - go_text.get_width() // 2, HEIGHT // 2),
                )

            pygame.display.flip()
            self.clock.tick(FPS)


def main():
    Game().run()


if __name__ == "__main__":
    main()
