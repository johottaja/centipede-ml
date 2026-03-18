"""
Pure game-logic engine for Centipede.
No event loop, no window – just state + step().
Rendering is done via render(surf) onto a caller-supplied pygame Surface.
"""
import random
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE = 16
COLS, ROWS = 30, 31
PLAYER_ZONE_TOP = ROWS - 5
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE

COLOR_BG = (0, 0, 0)
COLOR_MUSHROOM = [(34, 139, 34), (0, 100, 0), (0, 60, 0), (60, 40, 20)]
COLOR_PLAYER = (0, 200, 255)
COLOR_BULLET = (255, 255, 100)
COLOR_CENTIPEDE_HEAD = (255, 50, 50)
COLOR_CENTIPEDE_BODY = (200, 0, 0)
COLOR_HUD = (255, 255, 255)

MUSHROOM_HP = 4
CENTIPEDE_LENGTH = 12
CENTIPEDE_SPEED = 2
PLAYER_SPEED = 4
BULLET_SPEED = 8
SHOOT_COOLDOWN = 8


# ---------------------------------------------------------------------------
# Action constants  (used by the gym env and the human runner)
# ---------------------------------------------------------------------------
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_FIRE = 5
ACTION_LEFT_FIRE = 6
ACTION_RIGHT_FIRE = 7
ACTION_UP_FIRE = 8
ACTION_DOWN_FIRE = 9
NUM_ACTIONS = 10


# ---------------------------------------------------------------------------
# Internal game objects
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
        shrink = stage * 2
        inner = r.inflate(-shrink, -shrink)
        pygame.draw.ellipse(surf, color, inner)

    def hit(self):
        self.hp -= 1
        return self.hp <= 0


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


class Segment:
    def __init__(self, x, y, is_head=False):
        self.x = float(x)
        self.y = float(y)
        self.is_head = is_head
        self.dir = 1
        self.dropping = 0
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
            ex = r.centerx + self.dir * 3
            pygame.draw.circle(surf, (255, 255, 255), (ex, r.centery - 2), 2)
            pygame.draw.circle(surf, (255, 255, 255), (ex, r.centery + 2), 2)


class Centipede:
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
            hit_mush = not hit_wall and field.get(next_col, seg.row) is not None

            if hit_wall or hit_mush:
                seg.x = seg.col * TILE
                seg.dropping = TILE

    def draw(self, surf):
        for seg in reversed(self.segments):
            seg.draw(surf)


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


class Player:
    def __init__(self):
        self.x = WIDTH // 2 - TILE // 2
        self.y = (ROWS - 2) * TILE
        self.cooldown = 0
        self.lives = 3

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, TILE, TILE)

    def apply_action(self, action: int):
        """Move and optionally fire based on a discrete action integer."""
        move = action in (ACTION_LEFT, ACTION_LEFT_FIRE)
        if move:
            self.x = max(0, self.x - PLAYER_SPEED)
        move = action in (ACTION_RIGHT, ACTION_RIGHT_FIRE)
        if move:
            self.x = min(WIDTH - TILE, self.x + PLAYER_SPEED)
        move = action in (ACTION_UP, ACTION_UP_FIRE)
        if move:
            self.y = max(PLAYER_ZONE_TOP * TILE, self.y - PLAYER_SPEED)
        move = action in (ACTION_DOWN, ACTION_DOWN_FIRE)
        if move:
            self.y = min(HEIGHT - TILE, self.y + PLAYER_SPEED)
        if self.cooldown > 0:
            self.cooldown -= 1

    def wants_fire(self, action: int) -> bool:
        return action in (ACTION_FIRE, ACTION_LEFT_FIRE, ACTION_RIGHT_FIRE,
                          ACTION_UP_FIRE, ACTION_DOWN_FIRE)

    def shoot(self):
        if self.cooldown > 0:
            return None
        self.cooldown = SHOOT_COOLDOWN
        return Bullet(self.x + TILE // 2, self.y)

    def draw(self, surf):
        r = self.rect
        pts = [(r.centerx, r.top), (r.left, r.bottom), (r.right, r.bottom)]
        pygame.draw.polygon(surf, COLOR_PLAYER, pts)


# ---------------------------------------------------------------------------
# Engine  – headless, deterministic game state
# ---------------------------------------------------------------------------
class GameEngine:
    """
    Headless game engine.  Call reset() then step(action) in a loop.
    render(surf) draws the current state onto any pygame Surface.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._font = None  # initialised lazily so pygame.font is optional
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = random.Random(seed)

        # Temporarily swap the global random so MushroomField.populate uses
        # our seeded RNG.
        _orig = random.random
        random.random = self._rng.random  # type: ignore[assignment]

        self.score = 0
        self.player = Player()
        self.bullets: list[Bullet] = []
        self.centipedes: list[Centipede] = []
        self.field = MushroomField()
        self.field.populate()
        self._spawn_centipede()
        self.terminated = False
        self.truncated = False

        random.random = _orig  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def _spawn_centipede(self):
        start_x = (COLS // 2) * TILE
        segs = [Segment(start_x - i * TILE, 0, is_head=(i == 0))
                for i in range(CENTIPEDE_LENGTH)]
        self.centipedes.append(Centipede(segs))

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[int, bool, bool]:
        """
        Advance the game by one frame.

        Returns
        -------
        reward : int
        terminated : bool   – player lost all lives
        truncated : bool    – always False (no time limit enforced here)
        """
        if self.terminated:
            return 0, True, False

        reward = 0

        self.player.apply_action(action)
        if self.player.wants_fire(action) and not self.bullets:
            b = self.player.shoot()
            if b:
                self.bullets.append(b)

        for b in self.bullets:
            b.update()
        self.bullets = [b for b in self.bullets if b.alive]

        for c in self.centipedes:
            c.update(self.field)

        reward += self._handle_bullet_mushroom()
        reward += self._handle_bullet_centipede()
        self._handle_player_centipede()

        if not self.centipedes:
            self._spawn_centipede()

        self.score += reward
        return reward, self.terminated, self.truncated

    # ------------------------------------------------------------------
    def _handle_bullet_mushroom(self) -> int:
        reward = 0
        for b in self.bullets:
            if not b.alive:
                continue
            m = self.field.collides(b.rect)
            if m:
                b.alive = False
                if m.hit():
                    self.field.remove(m.col, m.row)
                    reward += 5
                else:
                    reward += 1
        return reward

    def _handle_bullet_centipede(self) -> int:
        reward = 0
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
                reward += 100 if seg.is_head else 10

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
        return reward

    def _handle_player_centipede(self):
        for chain in self.centipedes:
            for seg in chain.segments:
                if self.player.rect.colliderect(seg.rect):
                    self.player.lives -= 1
                    if self.player.lives <= 0:
                        self.terminated = True
                    else:
                        self.player.x = WIDTH // 2 - TILE // 2
                        self.player.y = (ROWS - 2) * TILE
                    return

    # ------------------------------------------------------------------
    def render(self, surf: pygame.Surface, font: pygame.font.Font | None = None):
        surf.fill(COLOR_BG)
        self.field.draw(surf)
        for c in self.centipedes:
            c.draw(surf)
        for b in self.bullets:
            b.draw(surf)
        if not self.terminated:
            self.player.draw(surf)

        if font:
            hud = font.render(
                f"Score: {self.score}   Lives: {self.player.lives}", True, COLOR_HUD
            )
            surf.blit(hud, (8, HEIGHT - 20))
            if self.terminated:
                msg = font.render("GAME OVER – press R to restart", True, COLOR_HUD)
                surf.blit(msg, (surf.get_width() // 2 - msg.get_width() // 2,
                                surf.get_height() // 2))
