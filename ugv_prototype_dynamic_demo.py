
import pygame
import heapq
import math
import random

# ------------------------------------------------------------
# GPS-DENIED UGV - 1 DAY SOFTWARE PROTOTYPE
# ------------------------------------------------------------
# Controls:
#   R     = reset simulation
#   P     = pause/resume
#   ESC   = quit
#
# The L-shaped dynamic obstacle moves automatically.
# When it interferes with the planned route, the UGV replans.
#
# This is a simulation prototype. The "perception" layer is
# represented by the virtual environment's obstacle grid.
# The architecture can later replace that input with camera,
# depth, SLAM and AI outputs.
# ------------------------------------------------------------

pygame.init()
pygame.display.set_caption("GPS-Denied UGV - Risk-Aware Navigation Prototype")

W, H = 1200, 720
PANEL_W = 330
MAP_W = W - PANEL_W
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

# Colors
BG = (18, 22, 28)
PANEL = (27, 32, 40)
GRID = (45, 51, 61)
FREE = (42, 58, 48)
RISK = (104, 82, 39)
OBSTACLE = (150, 58, 58)
PATH = (70, 190, 210)
BACKUP = (125, 105, 190)
UGV = (80, 180, 100)
GOAL = (240, 190, 60)
TEXT = (235, 238, 242)
MUTED = (160, 170, 182)
WHITE = (255, 255, 255)

ROWS, COLS = 24, 29
CELL = min((MAP_W - 50) // COLS, (H - 80) // ROWS)
OX = 25
OY = 45

START = (2, 2)
GOAL_POS = (ROWS - 3, COLS - 3)

# Fixed obstacles make the demo repeatable.
FIXED_OBS = {
    (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
    (8, 8), (8, 9), (8, 10),
    (13, 5), (14, 5), (15, 5), (16, 5),
    (16, 6), (16, 7),
    (5, 18), (6, 18), (7, 18), (8, 18),
    (9, 18), (9, 19), (9, 20), (9, 21),
    (17, 14), (17, 15), (17, 16), (18, 16), (19, 16),
}

DYNAMIC_OBS = set()

# Four moving L-shaped obstacles.
# Each one has a different patrol line and speed so the environment
# changes continuously and the UGV must react to moving hazards.
DYNAMIC_SHAPES = [
    {"origin": (3, 10),  "dr": 0,  "dc": 1,  "speed": 4.2, "timer": 0.0},
    {"origin": (6, 18),  "dr": 1,  "dc": 0,  "speed": 3.8, "timer": 0.0},
    {"origin": (11, 23), "dr": 0,  "dc": -1, "speed": 4.0, "timer": 0.0},
    {"origin": (16, 8),  "dr": -1, "dc": 0, "speed": 3.6, "timer": 0.0},
]

dynamic_move_interval = 0.08
replan_cooldown = 0.0

paused = False
mission_complete = False
event_message = "MISSION STARTED"
event_timer = 2.5
risk_score = 18
confidence = 0.92
speed_state = "NORMAL"
decision = "CONTINUE"
reason = "Current route has low predicted risk."
last_replanned = False

def neighbors(node):
    r, c = node
    for dr, dc, cost in [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
        (-1, -1, 1.414), (-1, 1, 1.414),
        (1, -1, 1.414), (1, 1, 1.414)
    ]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield (nr, nc), cost

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def near_obstacle(node, obstacles, radius=1):
    r, c = node
    for rr in range(r-radius, r+radius+1):
        for cc in range(c-radius, c+radius+1):
            if (rr, cc) in obstacles:
                return True
    return False

def risk_cost(node, obstacles):
    # Higher cost near obstacles, representing a dynamic safety margin.
    if node in obstacles:
        return 9999
    r, c = node
    min_d = 99
    for o in obstacles:
        d = math.hypot(r-o[0], c-o[1])
        min_d = min(min_d, d)
    if min_d <= 1.1:
        return 9
    if min_d <= 2.1:
        return 4
    if min_d <= 3.1:
        return 1.5
    return 0

def astar(start, goal, obstacles):
    pq = [(0, start)]
    came = {}
    g = {start: 0}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        for nxt, step in neighbors(cur):
            if nxt in obstacles:
                continue
            new_g = g[cur] + step + risk_cost(nxt, obstacles)
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came[nxt] = cur
                f = new_g + heuristic(nxt, goal)
                heapq.heappush(pq, (f, nxt))
    return []

def path_risk(path, obstacles):
    if not path:
        return 100
    score = 10
    for p in path:
        if p in obstacles:
            return 100
        if near_obstacle(p, obstacles, 1):
            score += 2.5
        if near_obstacle(p, obstacles, 2):
            score += 0.8
    return min(99, int(score))

def cell_center(node):
    r, c = node
    return (OX + c * CELL + CELL // 2, OY + r * CELL + CELL // 2)

def update_dynamic_obstacle(dt):
    global DYNAMIC_OBS, DYNAMIC_SHAPES, replan_cooldown

    shape = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    DYNAMIC_OBS = set()
    replan_cooldown = max(0.0, replan_cooldown - dt)

    for i, obj in enumerate(DYNAMIC_SHAPES):
        obj["timer"] += dt

        # Continuous movement instead of jumping cell-to-cell.
        if obj["timer"] >= dynamic_move_interval:
            step = obj["timer"] * obj["speed"]
            obj["timer"] = 0.0

            r, c = obj["origin"]
            nr = r + obj["dr"] * step
            nc = c + obj["dc"] * step

            # Each obstacle sweeps a corridor that intersects the
            # UGV's normal route at different points.
            if i == 0:          # upper horizontal sweeper
                if nc >= 21:
                    nc = 21
                    obj["dc"] = -1
                elif nc <= 7:
                    nc = 7
                    obj["dc"] = 1

            elif i == 1:        # upper/right vertical sweeper
                if nr >= 15:
                    nr = 15
                    obj["dr"] = -1
                elif nr <= 4:
                    nr = 4
                    obj["dr"] = 1

            elif i == 2:        # right horizontal sweeper
                if nc <= 10:
                    nc = 10
                    obj["dc"] = 1
                elif nc >= 24:
                    nc = 24
                    obj["dc"] = -1

            else:               # lower vertical sweeper
                if nr <= 9:
                    nr = 9
                    obj["dr"] = 1
                elif nr >= 19:
                    nr = 19
                    obj["dr"] = -1

            obj["origin"] = (nr, nc)

        r = int(round(obj["origin"][0]))
        c = int(round(obj["origin"][1]))

        for sr, sc in shape:
            nr, nc = r + sr, c + sc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                DYNAMIC_OBS.add((nr, nc))



def reset():
    global DYNAMIC_OBS, paused, mission_complete, event_message
    global event_timer, risk_score, confidence, speed_state, decision, reason
    global DYNAMIC_SHAPES, replan_cooldown

    DYNAMIC_OBS = set()
    DYNAMIC_SHAPES = [
        {"origin": (3.0, 10.0),  "dr": 0,  "dc": 1,  "speed": 4.2, "timer": 0.0},
        {"origin": (6.0, 18.0),  "dr": 1,  "dc": 0,  "speed": 3.8, "timer": 0.0},
        {"origin": (11.0, 23.0), "dr": 0,  "dc": -1, "speed": 4.0, "timer": 0.0},
        {"origin": (16.0, 8.0),  "dr": -1, "dc": 0,  "speed": 3.6, "timer": 0.0},
    ]
    replan_cooldown = 0.0

    paused = False
    mission_complete = False
    event_message = "MISSION STARTED"
    event_timer = 2.5
    risk_score = 18
    confidence = 0.92
    speed_state = "NORMAL"
    decision = "CONTINUE"
    reason = "Current route has low predicted risk."

reset()
update_dynamic_obstacle(0.5)

update_dynamic_obstacle(999)

obstacles = set(FIXED_OBS) | set(DYNAMIC_OBS)
path = astar(START, GOAL_POS, obstacles)
backup_path = path[:]
ugv_index = 0
ugv_pos = START
move_timer = 0

font = pygame.font.SysFont("arial", 19)
small = pygame.font.SysFont("arial", 16)
title = pygame.font.SysFont("arial", 25, bold=True)
big = pygame.font.SysFont("arial", 30, bold=True)

def draw_text(txt, x, y, f=font, color=TEXT):
    screen.blit(f.render(txt, True, color), (x, y))

running = True
while running:
    dt = clock.tick(60) / 1000.0

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_r:
                reset()
                update_dynamic_obstacle(0.5)
                obstacles = set(FIXED_OBS) | set(DYNAMIC_OBS)
                path = astar(START, GOAL_POS, obstacles)
                backup_path = path[:]
                ugv_index = 0
                ugv_pos = START
            elif e.key == pygame.K_p:
                paused = not paused

    if not paused and not mission_complete:
        update_dynamic_obstacle(dt)

    obstacles = set(FIXED_OBS) | set(DYNAMIC_OBS)

    # Look ahead instead of waiting for an exact collision.
    # This demonstrates predictive perception + safety-margin behavior.
    lookahead = path[max(ugv_index, 0): min(len(path), ugv_index + 14)]
    dynamic_threat = any(
        near_obstacle(p, DYNAMIC_OBS, radius=2)
        for p in lookahead
    )

    if path and dynamic_threat and replan_cooldown <= 0:
        new_path = astar(ugv_pos, GOAL_POS, obstacles)

        if new_path and new_path != path:
            path = new_path
            ugv_index = 0
            last_replanned = True
            replan_cooldown = 1.2

            decision = "REPLAN"
            risk_score = max(72, path_risk(path, obstacles))
            confidence = 0.84
            speed_state = "CAUTIOUS"
            reason = "Moving obstacle detected ahead; safer route selected."
            event_message = "DYNAMIC OBSTACLE DETECTED — REPLANNING"
            event_timer = 2.5


    if not paused and not mission_complete:
        # Simulated motion.
        move_timer += dt
        if move_timer >= 0.075:
            move_timer = 0
            if ugv_index < len(path) - 1:
                ugv_index += 1
                ugv_pos = path[ugv_index]
            else:
                mission_complete = True
                decision = "MISSION COMPLETE"
                speed_state = "STOPPED"
                risk_score = 5
                reason = "Goal reached successfully."

    # If no active dynamic event, normal status.
    if not mission_complete and event_timer <= 0:
        if DYNAMIC_OBS:
            decision = "CAUTIOUS"
            speed_state = "CAUTIOUS"
            risk_score = max(35, path_risk(path, obstacles))
            confidence = 0.89
            reason = "Dynamic objects detected; monitoring the safety corridor."
        else:
            decision = "CONTINUE"
            speed_state = "NORMAL"
            risk_score = max(10, path_risk(path, obstacles))
            confidence = 0.92
            reason = "Current route has low predicted collision risk."

    if event_timer > 0:
        event_timer -= dt

    # ---------------- DRAW MAP ----------------
    screen.fill(BG)

    # Header
    draw_text("GPS-DENIED UGV — RISK-AWARE NAVIGATION PROTOTYPE",
              25, 12, title)

    # Map background
    pygame.draw.rect(screen, (22, 27, 33),
                     (0, 40, MAP_W, H-40))

    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(OX + c*CELL, OY + r*CELL, CELL-1, CELL-1)
            node = (r, c)
            if node in obstacles:
                pygame.draw.rect(screen, OBSTACLE, rect)
            elif node in path:
                pygame.draw.rect(screen, (31, 67, 70), rect)
            else:
                pygame.draw.rect(screen, FREE, rect)

            pygame.draw.rect(screen, GRID, rect, 1)

    # Backup-ish / current path
    if len(path) > 1:
        pts = [cell_center(p) for p in path]
        pygame.draw.lines(screen, PATH, False, pts, 4)

    # Start
    pygame.draw.circle(screen, (80, 140, 220), cell_center(START),
                       max(7, CELL//3))
    draw_text("START", cell_center(START)[0]-25,
              cell_center(START)[1]-35, small)

    # Goal
    pygame.draw.circle(screen, GOAL, cell_center(GOAL_POS),
                       max(8, CELL//3))
    draw_text("GOAL", cell_center(GOAL_POS)[0]-20,
              cell_center(GOAL_POS)[1]-35, small)

    # UGV
    ux, uy = cell_center(ugv_pos)
    pygame.draw.circle(screen, UGV, (ux, uy), max(9, CELL//2-2))
    pygame.draw.circle(screen, WHITE, (ux, uy), max(9, CELL//2-2), 2)

    # Safety bubble
    bubble = int(CELL * (2.2 if confidence < 0.8 else 1.5))
    pygame.draw.circle(screen, (90, 180, 110), (ux, uy),
                       bubble, 1)

    # Dynamic obstacle highlight
    for d in DYNAMIC_OBS:
        dx, dy = cell_center(d)
        pygame.draw.circle(screen, (255, 80, 80), (dx, dy),
                           max(8, CELL//2-2), 2)

    # ---------------- DRAW PANEL ----------------
    pygame.draw.rect(screen, PANEL, (MAP_W, 0, PANEL_W, H))
    pygame.draw.line(screen, GRID, (MAP_W, 0), (MAP_W, H), 2)

    x = MAP_W + 20
    y = 28

    draw_text("SYSTEM STATUS", x, y, title)
    y += 48

    draw_text("UGV POSITION", x, y, small, MUTED)
    y += 23
    draw_text(f"Grid: ({ugv_pos[0]}, {ugv_pos[1]})", x, y)
    y += 35

    draw_text("GOAL", x, y, small, MUTED)
    y += 23
    draw_text(f"Grid: ({GOAL_POS[0]}, {GOAL_POS[1]})", x, y)
    y += 35

    draw_text("PERCEPTION", x, y, small, MUTED)
    y += 23
    draw_text(f"Obstacles: {len(obstacles)}", x, y)
    y += 24
    draw_text("Terrain: 68% traversable", x, y)
    y += 35

    draw_text("RISK ENGINE", x, y, small, MUTED)
    y += 24
    draw_text(f"Risk score: {risk_score}/100", x, y)
    y += 24
    draw_text(f"Safety margin: {'LARGE' if confidence < 0.8 else 'NORMAL'}",
              x, y)
    y += 35

    draw_text("LOCALIZATION", x, y, small, MUTED)
    y += 24
    draw_text(f"Visual confidence: {int(confidence*100)}%", x, y)
    y += 35

    draw_text("DECISION", x, y, small, MUTED)
    y += 26
    draw_text(decision, x, y, big,
              (240, 205, 80) if decision != "MISSION COMPLETE" else GOAL)
    y += 45

    draw_text("REASON", x, y, small, MUTED)
    y += 24

    # Wrap reason
    words = reason.split()
    line = ""
    for word in words:
        test = line + (" " if line else "") + word
        if small.size(test)[0] > PANEL_W - 40:
            draw_text(line, x, y, small)
            y += 21
            line = word
        else:
            line = test
    if line:
        draw_text(line, x, y, small)
        y += 35

    draw_text("CONTROLS", x, y, small, MUTED)
    y += 23
    draw_text("AUTO   Moving obstacle", x, y, small)
    y += 21
    draw_text("R      Reset", x, y, small)
    y += 21
    draw_text("P      Pause", x, y, small)
    y += 21
    draw_text("ESC    Quit", x, y, small)

    # Bottom event banner
    if event_timer > 0:
        banner = pygame.Rect(25, H-48, MAP_W-50, 34)
        pygame.draw.rect(screen, (38, 48, 55), banner, border_radius=6)
        draw_text(event_message, banner.x+12, banner.y+7, small)

    if paused:
        draw_text("PAUSED", MAP_W//2-55, 80, big, GOAL)

    pygame.display.flip()

pygame.quit()
