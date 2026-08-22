"""
Pure game-logic engine for Centipede.
No event loop, no window – just state + step().
Rendering is done via render(surf) onto a caller-supplied pygame Surface.
"""
import math
import random
import numpy as np
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
COLOR_SPIDER = (255, 165, 0)
COLOR_HUD = (255, 255, 255)

MUSHROOM_HP = 4
CENTIPEDE_LENGTH = 12
CENTIPEDE_SPEED = 2
PLAYER_SPEED = 4
BULLET_SPEED = 8
SHOOT_COOLDOWN = 8

# Spider constants
SPIDER_SPEED = 2
SPIDER_SPAWN_INTERVAL = 300   # frames between spider spawns
SPIDER_LIFETIME = 600         # frames before a spider despawns on its own
SPIDER_DIR_CHANGE_INTERVAL = 30  # frames between random direction changes


# ---------------------------------------------------------------------------
# Action constants  (used by the gym env and the human runner)
# ---------------------------------------------------------------------------
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_FIRE = 5
NUM_ACTIONS = 6


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
                if m:
                    return m
        return None


class Segment:
    def __init__(self, x, y, is_head=False):
        self.x = float(x)
        self.y = float(y)
        self.is_head = is_head
        self.dir = 1
        self.dropping = 0
        self.vdir = 1  # vertical direction: +1 = downward, -1 = upward
        self.speed = CENTIPEDE_SPEED
        self._rect = pygame.Rect(int(x), int(y), TILE, TILE)

    def _sync_rect(self):
        self._rect.x = int(self.x)
        self._rect.y = int(self.y)

    @property
    def rect(self):
        self._sync_rect()
        return self._rect

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
                seg.y += seg.vdir * dy
                seg.dropping -= dy
                if seg.dropping == 0:
                    # Clamp to grid row boundary after drop completes
                    seg.y = seg.row * TILE
                    # Reverse vertical direction at top (row 0) or bottom (ROWS-1)
                    if seg.row <= 0 or seg.row >= ROWS - 1:
                        seg.vdir *= -1
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


class Spider:
    """
    Erratic enemy that roams the player zone, eating mushrooms it touches.
    Spawns at a random side edge of the player zone and wanders until killed
    or its lifetime expires.
    """

    def __init__(self, rng: random.Random):
        # Spawn on a random side, in the player zone rows
        side = rng.choice((-1, 1))
        self.x = float(0 if side == 1 else WIDTH - TILE)
        self.y = float(rng.randint(PLAYER_ZONE_TOP, ROWS - 2) * TILE)
        self.dx = float(side * SPIDER_SPEED)
        self.dy = float(rng.choice((-1, 0, 1)) * SPIDER_SPEED)
        self.alive = True
        self.lifetime = SPIDER_LIFETIME
        self._dir_timer = 0
        self._rng = rng

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), TILE, TILE)

    @property
    def col(self):
        return int(self.x) // TILE

    @property
    def row(self):
        return int(self.y) // TILE

    def update(self, field: "MushroomField"):
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False
            return

        self._dir_timer += 1
        if self._dir_timer >= SPIDER_DIR_CHANGE_INTERVAL:
            self._dir_timer = 0
            self.dx = float(self._rng.choice((-1, 0, 1)) * SPIDER_SPEED)
            self.dy = float(self._rng.choice((-1, 0, 1)) * SPIDER_SPEED)
            # Bias toward staying on-screen horizontally
            if self.x <= 0:
                self.dx = abs(self.dx) if self.dx == 0 else abs(self.dx)
            elif self.x >= WIDTH - TILE:
                self.dx = -abs(self.dx) if self.dx == 0 else -abs(self.dx)

        self.x = max(0.0, min(float(WIDTH - TILE), self.x + self.dx))
        self.y = max(float(PLAYER_ZONE_TOP * TILE),
                     min(float((ROWS - 1) * TILE), self.y + self.dy))

        # Eat any mushroom the spider overlaps
        m = field.collides(self.rect)
        if m:
            field.remove(m.col, m.row)

    def draw(self, surf: pygame.Surface):
        r = self.rect
        cx, cy = r.centerx, r.centery
        half = TILE // 2 - 1
        # Body: small filled circle
        pygame.draw.circle(surf, COLOR_SPIDER, (cx, cy), half - 2)
        # Eight legs as short lines radiating outward
        for angle_idx in range(8):
            angle = angle_idx * math.pi / 4
            lx = int(cx + math.cos(angle) * (half + 2))
            ly = int(cy + math.sin(angle) * (half + 2))
            pygame.draw.line(surf, COLOR_SPIDER, (cx, cy), (lx, ly), 1)


class SpiderManager:
    def __init__(self):
        self.spiders: list[Spider] = []
        self._spawn_timer = 0

    def reset(self):
        self.spiders = []
        self._spawn_timer = 0

    def update(self, field: "MushroomField", rng: random.Random):
        self._spawn_timer += 1
        if self._spawn_timer >= SPIDER_SPAWN_INTERVAL:
            self._spawn_timer = 0
            self.spiders.append(Spider(rng))

        for s in self.spiders:
            s.update(field)
        self.spiders = [s for s in self.spiders if s.alive]

    def draw(self, surf: pygame.Surface):
        for s in self.spiders:
            s.draw(surf)


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
        if action == ACTION_LEFT:
            self.x = max(0, self.x - PLAYER_SPEED)
        if action == ACTION_RIGHT:
            self.x = min(WIDTH - TILE, self.x + PLAYER_SPEED)
        if action == ACTION_UP:
            self.y = max(PLAYER_ZONE_TOP * TILE, self.y - PLAYER_SPEED)
        if action == ACTION_DOWN:
            self.y = min(HEIGHT - TILE, self.y + PLAYER_SPEED)
        if self.cooldown > 0:
            self.cooldown -= 1

    def wants_fire(self, action: int) -> bool:
        return action == ACTION_FIRE

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

    def __init__(
        self,
        seed: int | None = None,
        reward_mushroom_hit: int = 1,
        reward_mushroom_destroy: int = 5,
        reward_body_hit: int = 10,
        reward_head_hit: int = 100,
        reward_depth_discount: float = 0.0,
        reward_depth_discount_fn: str = "linear",
        reward_spider_hit: int = 300,
        reward_spider_penalty: int = 1000,
        reward_centipede_penalty: int = 1000,
        reward_survival: float = 0.01,
        reward_proximity_penalty: float = 1.0,
        proximity_distance_tiles: int = 3,
    ):
        self._rng = random.Random(seed)
        self._font = None  # initialised lazily so pygame.font is optional
        self.reward_mushroom_hit = reward_mushroom_hit
        self.reward_mushroom_destroy = reward_mushroom_destroy
        self.reward_body_hit = reward_body_hit
        self.reward_head_hit = reward_head_hit
        self.reward_depth_discount = reward_depth_discount
        self.reward_depth_discount_fn = reward_depth_discount_fn
        self.reward_spider_hit = reward_spider_hit
        self.reward_spider_penalty = reward_spider_penalty
        self.reward_centipede_penalty = reward_centipede_penalty
        self.reward_survival = reward_survival
        self.reward_proximity_penalty = reward_proximity_penalty
        self.proximity_distance_tiles = proximity_distance_tiles
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = random.Random(seed)

        # Temporarily swap the global random so MushroomField.populate uses
        # our seeded RNG.
        _orig = random.random
        random.random = self._rng.random  # type: ignore[assignment]

        self.score = 0.0
        self.segments_destroyed = 0
        self.spiders_destroyed = 0
        self.player = Player()
        self.bullets: list[Bullet] = []
        self.centipedes: list[Centipede] = []
        self.field = MushroomField()
        self.field.populate()
        self._spawn_centipede()
        self._spider_mgr = SpiderManager()
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
    def step(self, action: int) -> tuple[float, bool, bool]:
        """
        Advance the game by one frame.

        Returns
        -------
        reward : float
        terminated : bool   – player lost all lives
        truncated : bool    – always False (no time limit enforced here)
        """
        if self.terminated:
            return 0, True, False

        reward = 0.0

        self.player.apply_action(action)
        if self.player.wants_fire(action) and not self.bullets:
            b = self.player.shoot()
            if b:
                self.bullets.append(b)

        for b in self.bullets:
            b.update()
        if self.bullets and not self.bullets[0].alive:
            self.bullets.clear()

        for c in self.centipedes:
            c.update(self.field)

        self._spider_mgr.update(self.field, self._rng)

        reward += self._handle_bullet_mushroom()
        reward += self._handle_bullet_centipede()
        reward += self._handle_bullet_spider()
        reward += self._handle_player_centipede()
        reward += self._handle_player_spider()
        reward += self._proximity_penalty()
        reward += self.reward_survival

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
                    reward += self.reward_mushroom_destroy
                else:
                    reward += self.reward_mushroom_hit
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
                base = self.reward_head_hit if seg.is_head else self.reward_body_hit
                if self.reward_depth_discount > 0.0:
                    depth_fraction = seg.row / max(1, ROWS - 1)
                    if self.reward_depth_discount_fn == "exponential":
                        multiplier = (1.0 - self.reward_depth_discount) ** depth_fraction
                    else:  # linear
                        multiplier = 1.0 - self.reward_depth_discount * depth_fraction
                    reward += base * multiplier
                else:
                    reward += base
                self.segments_destroyed += 1

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

    def _handle_player_centipede(self) -> int:
        for chain in self.centipedes:
            for seg in chain.segments:
                if self.player.rect.colliderect(seg.rect):
                    self.player.lives -= 1
                    if self.player.lives <= 0:
                        self.terminated = True
                    else:
                        self.player.x = WIDTH // 2 - TILE // 2
                        self.player.y = (ROWS - 2) * TILE
                    return -self.reward_centipede_penalty
        return 0

    def _handle_bullet_spider(self) -> int:
        reward = 0
        for b in self.bullets:
            if not b.alive:
                continue
            for s in self._spider_mgr.spiders:
                if s.alive and b.rect.colliderect(s.rect):
                    b.alive = False
                    s.alive = False
                    self.spiders_destroyed += 1
                    reward += self.reward_spider_hit
                    break
        self._spider_mgr.spiders = [s for s in self._spider_mgr.spiders if s.alive]
        return reward

    def _handle_player_spider(self) -> int:
        for s in self._spider_mgr.spiders:
            if s.alive and self.player.rect.colliderect(s.rect):
                s.alive = False
                self._spider_mgr.spiders = [sp for sp in self._spider_mgr.spiders if sp.alive]
                self.player.lives -= 1
                if self.player.lives <= 0:
                    self.terminated = True
                else:
                    self.player.x = WIDTH // 2 - TILE // 2
                    self.player.y = (ROWS - 2) * TILE
                return -self.reward_spider_penalty
        return 0

    def _proximity_penalty(self) -> float:
        """Scaled penalty when any threat is within proximity_distance_tiles."""
        if self.reward_proximity_penalty <= 0.0 or self.proximity_distance_tiles <= 0:
            return 0.0

        px = self.player.x + TILE / 2
        py = self.player.y + TILE / 2
        threshold = self.proximity_distance_tiles * TILE
        worst = 0.0

        for chain in self.centipedes:
            for seg in chain.segments:
                d = math.hypot(seg.x + TILE / 2 - px, seg.y + TILE / 2 - py)
                if d < threshold:
                    factor = 1.0 - d / threshold
                    worst = max(worst, self.reward_proximity_penalty * factor)

        for s in self._spider_mgr.spiders:
            if not s.alive:
                continue
            d = math.hypot(s.x + TILE / 2 - px, s.y + TILE / 2 - py)
            if d < threshold:
                factor = 1.0 - d / threshold
                worst = max(worst, self.reward_proximity_penalty * factor)

        return -worst

    # ------------------------------------------------------------------
    # Occupancy-grid observation (5 channels, float32 in [0, 1])
    #   0 = player   1 = mushrooms   2 = centipede heads
    #   3 = centipede body   4 = spiders
    OCCUPANCY_CHANNELS = 5
    CH_PLAYER = 0
    CH_MUSHROOM = 1
    CH_HEAD = 2
    CH_BODY = 3
    CH_SPIDER = 4
    _TILE_AREA = float(TILE * TILE)

    @staticmethod
    def _stamp_rect_occupancy(
        channel: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> None:
        """Add fractional tile occupancy for an axis-aligned pixel rect."""
        c0 = max(0, left // TILE)
        c1 = min(COLS - 1, max(0, right - 1) // TILE)
        r0 = max(0, top // TILE)
        r1 = min(ROWS - 1, max(0, bottom - 1) // TILE)
        for row in range(r0, r1 + 1):
            tile_top = row * TILE
            tile_bottom = tile_top + TILE
            oy0 = max(top, tile_top)
            oy1 = min(bottom, tile_bottom)
            overlap_h = oy1 - oy0
            if overlap_h <= 0:
                continue
            for col in range(c0, c1 + 1):
                tile_left = col * TILE
                tile_right = tile_left + TILE
                ox0 = max(left, tile_left)
                ox1 = min(right, tile_right)
                overlap_w = ox1 - ox0
                if overlap_w > 0:
                    channel[row, col] += (overlap_w * overlap_h) / GameEngine._TILE_AREA

    def get_occupancy_obs(self, out: np.ndarray | None = None) -> np.ndarray:
        """Build a 5-channel occupancy grid, shape (5, ROWS, COLS), float32 in [0, 1].

        Each channel scores tiles by the fraction of the tile area covered by
        that entity type. Values are clipped to 1.0 per channel.
        """
        if out is None:
            out = np.zeros((self.OCCUPANCY_CHANNELS, ROWS, COLS), dtype=np.float32)
        else:
            out[:] = 0.0

        for m in self.field.grid.values():
            out[self.CH_MUSHROOM, m.row, m.col] = 1.0

        for chain in self.centipedes:
            for seg in chain.segments:
                r = seg.rect
                ch = self.CH_HEAD if seg.is_head else self.CH_BODY
                self._stamp_rect_occupancy(out[ch], r.left, r.top, r.right, r.bottom)

        for s in self._spider_mgr.spiders:
            if s.alive:
                r = s.rect
                self._stamp_rect_occupancy(
                    out[self.CH_SPIDER], r.left, r.top, r.right, r.bottom
                )

        if not self.terminated:
            r = self.player.rect
            self._stamp_rect_occupancy(
                out[self.CH_PLAYER], r.left, r.top, r.right, r.bottom
            )

        np.clip(out, 0.0, 1.0, out=out)
        return out

    # ------------------------------------------------------------------
    # Grid observation encoding (legacy discrete grid)
    # 0=empty  1=mushroom  2=centipede body  3=centipede head
    # 4=player  5=bullet  6=spider
    GRID_EMPTY = 0
    GRID_MUSHROOM = 1
    GRID_BODY = 2
    GRID_HEAD = 3
    GRID_PLAYER = 4
    GRID_BULLET = 5
    GRID_SPIDER = 6
    GRID_MAX = 6  # highest value in the encoding (used by env observation_space)

    # ------------------------------------------------------------------
    # Relative / entity-centric observation
    #
    # Layout (105 float32 values):
    #   [0  .. 83]  12 centipede segment slots × 7 features each:
    #                rel_x, rel_y, vel_x, vel_y, is_alive, is_head, dist_to_obstacle
    #   [84 .. 86]  bullet: rel_x, rel_y, is_alive
    #   [87 .. 96]  2 spider slots × 5 features each:
    #                rel_x, rel_y, vel_x, vel_y, is_alive
    #   [97 .. 104] 8-way lidar distances from the player
    #               (walls, mushrooms, centipede segments, spiders)
    #               order: N, NE, E, SE, S, SW, W, NW
    _SEG_FEATURES = 7
    _SPIDER_FEATURES = 5
    # SPIDER_LIFETIME / SPIDER_SPAWN_INTERVAL = 600 / 300 → at most 2 alive at once
    _MAX_SPIDERS = 2
    _LIDAR_DIRS: list[tuple[int, int]] = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0,  1), (-1, 1), (-1, 0), (-1, -1),
    ]
    RELATIVE_OBS_SIZE = (
        CENTIPEDE_LENGTH * _SEG_FEATURES          # 84
        + 3                                        # bullet
        + _MAX_SPIDERS * _SPIDER_FEATURES          # 10
        + len(_LIDAR_DIRS)                         # 8
    )  # 105

    def _seg_obstacle_dist(self, seg: "Segment") -> float:
        """Horizontal distance in tiles to the next wall/mushroom in seg.dir.

        Returns 0.0 while the segment is dropping (moving vertically).
        Normalised by COLS so the result is in (0, 1].
        """
        if seg.dropping > 0:
            return 0.0
        col = seg.col
        row = seg.row
        d = seg.dir  # +1 or -1
        steps = 0
        c = col
        while steps < COLS:
            c += d
            steps += 1
            if c < 0 or c >= COLS:
                break
            if self.field.get(c, row) is not None:
                break
        return steps / float(COLS)

    def _tile_blocked(self, col: int, row: int) -> bool:
        """Return True if a lidar ray is blocked at (col, row)."""
        if self.field.get(col, row) is not None:
            return True
        for chain in self.centipedes:
            for seg in chain.segments:
                if seg.col == col and seg.row == row:
                    return True
        for s in self._spider_mgr.spiders:
            if s.alive and s.col == col and s.row == row:
                return True
        return False

    def get_relative_obs(self, out: np.ndarray | None = None) -> np.ndarray:
        """Build the entity-centric feature vector and return it.

        *out* must be a contiguous float32 array of length RELATIVE_OBS_SIZE
        when provided; passing a pre-allocated buffer avoids heap allocation.
        """
        if out is None:
            out = np.zeros(self.RELATIVE_OBS_SIZE, dtype=np.float32)
        else:
            out[:] = 0.0

        px = float(self.player.x)
        py = float(self.player.y)
        pcx = px + TILE / 2
        pcy = py + TILE / 2
        pcol = int(self.player.x) // TILE
        prow = int(self.player.y) // TILE

        # Collect every live segment, nearest-first (total ≤ CENTIPEDE_LENGTH)
        all_segs: list[Segment] = []
        for chain in self.centipedes:
            all_segs.extend(chain.segments)
        all_segs.sort(
            key=lambda seg: math.hypot(seg.x + TILE / 2 - pcx, seg.y + TILE / 2 - pcy)
        )

        offset = 0
        for i in range(CENTIPEDE_LENGTH):
            if i < len(all_segs):
                seg = all_segs[i]
                rel_x = (seg.x - px) / WIDTH
                rel_y = (seg.y - py) / HEIGHT
                if seg.dropping > 0:
                    vel_x = 0.0
                    vel_y = float(seg.vdir)   # ±1 while falling
                else:
                    vel_x = float(seg.dir)    # ±1 while traversing
                    vel_y = 0.0
                out[offset]     = rel_x
                out[offset + 1] = rel_y
                out[offset + 2] = vel_x
                out[offset + 3] = vel_y
                out[offset + 4] = 1.0                             # is_alive
                out[offset + 5] = 1.0 if seg.is_head else 0.0    # is_head
                out[offset + 6] = self._seg_obstacle_dist(seg)
            # unoccupied slot stays all-zero (is_alive = 0)
            offset += self._SEG_FEATURES

        # Bullet
        if self.bullets:
            b = self.bullets[0]
            out[offset]     = (b.x - px) / WIDTH
            out[offset + 1] = (b.y - py) / HEIGHT
            out[offset + 2] = 1.0  # is_alive
        offset += 3

        # Spider slots (nearest-first)
        spiders = sorted(
            self._spider_mgr.spiders,
            key=lambda s: math.hypot(s.x + TILE / 2 - pcx, s.y + TILE / 2 - pcy),
        )
        for i in range(self._MAX_SPIDERS):
            if i < len(spiders):
                s = spiders[i]
                out[offset]     = (s.x - px) / WIDTH
                out[offset + 1] = (s.y - py) / HEIGHT
                out[offset + 2] = s.dx / max(abs(s.dx), 1e-6) if s.dx != 0 else 0.0  # sign of dx
                out[offset + 3] = s.dy / max(abs(s.dy), 1e-6) if s.dy != 0 else 0.0  # sign of dy
                out[offset + 4] = 1.0  # is_alive
            # unoccupied slot stays all-zero
            offset += self._SPIDER_FEATURES

        # 8-way lidar from player tile (walls, mushrooms, centipede, spiders)
        max_dist = float(max(COLS, ROWS))
        for d_idx, (dc, dr) in enumerate(self._LIDAR_DIRS):
            c, r = pcol, prow
            steps = 0
            while True:
                c += dc
                r += dr
                steps += 1
                if c < 0 or c >= COLS or r < 0 or r >= ROWS:
                    break
                if self._tile_blocked(c, r):
                    break
            out[offset + d_idx] = steps / max_dist

        return out

    def get_grid_obs(self, out: np.ndarray | None = None) -> np.ndarray:
        """Write the game grid into *out* (or a fresh array) and return it.

        *out* must be a contiguous uint8 array of length COLS*ROWS when provided.
        Passing a pre-allocated buffer avoids a heap allocation on every step.
        """
        if out is None:
            out = np.empty(COLS * ROWS, dtype=np.uint8)
        out[:] = self.GRID_EMPTY

        for m in self.field.grid.values():
            out[m.row * COLS + m.col] = self.GRID_MUSHROOM

        for chain in self.centipedes:
            for seg in chain.segments:
                col = int(seg.x) // TILE
                row = int(seg.y) // TILE
                idx = row * COLS + col
                if 0 <= idx < COLS * ROWS:
                    out[idx] = self.GRID_HEAD if seg.is_head else self.GRID_BODY

        for s in self._spider_mgr.spiders:
            s_col = int(s.x) // TILE
            s_row = int(s.y) // TILE
            if 0 <= s_col < COLS and 0 <= s_row < ROWS:
                out[s_row * COLS + s_col] = self.GRID_SPIDER

        for b in self.bullets:
            col = b.x // TILE
            row = b.y // TILE
            if 0 <= col < COLS and 0 <= row < ROWS:
                out[row * COLS + col] = self.GRID_BULLET

        p_col = self.player.x // TILE
        p_row = self.player.y // TILE
        if 0 <= p_col < COLS and 0 <= p_row < ROWS:
            out[p_row * COLS + p_col] = self.GRID_PLAYER

        return out

    # ------------------------------------------------------------------
    def render(self, surf: pygame.Surface, font: pygame.font.Font | None = None):
        surf.fill(COLOR_BG)
        self.field.draw(surf)
        for c in self.centipedes:
            c.draw(surf)
        self._spider_mgr.draw(surf)
        for b in self.bullets:
            b.draw(surf)
        if not self.terminated:
            self.player.draw(surf)

        if font:
            hud = font.render(
                f"Score: {self.score:.0f}   Lives: {self.player.lives}", True, COLOR_HUD
            )
            surf.blit(hud, (8, HEIGHT - 20))
            if self.terminated:
                msg = font.render("GAME OVER – press R to restart", True, COLOR_HUD)
                surf.blit(msg, (surf.get_width() // 2 - msg.get_width() // 2,
                                surf.get_height() // 2))
