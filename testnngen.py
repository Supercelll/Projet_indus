import tcod as libtcod
from random import randint, random
from math import sqrt
from time import sleep
import numpy as np
import heapq
from nn2 import *
# actual size of the window
SCREEN_WIDTH = 60
SCREEN_HEIGHT = 35

# size of the map
MAP_WIDTH = 18   # width 18 pour que la matrice affichée en console corresponde visuellement à la map
MAP_HEIGHT = 18

# parameters for dungeon generator
ROOM_MAX_SIZE = 8
ROOM_MIN_SIZE = 5
MAX_ROOMS = 2


FOV_ALGO = 0  # default FOV algorithm
FOV_LIGHT_WALLS = True  # light walls or not
TORCH_RADIUS = 10

LIMIT_FPS = 20  # 20 frames-per-second maximum


color_dark_wall = libtcod.Color(0, 0, 100)
color_light_wall = libtcod.Color(130, 110, 50)
color_dark_ground = libtcod.Color(50, 50, 150)
color_light_ground = libtcod.Color(200, 180, 50)


class Tile:
    # a tile of the map and its properties
    def __init__(self, blocked, block_sight=None):
        self.blocked = blocked

        # all tiles start unexplored
        self.explored = False

        # by default, if a tile is blocked, it also blocks sight
        if block_sight is None: block_sight = blocked
        self.block_sight = block_sight


class Rect:
    # a rectangle on the map. used to characterize a room.
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        self.size = w*h

    def center(self):
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return center_x, center_y

    def intersect(self, other):
        # returns true if this rectangle intersects with another one
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)


class Noeud:
    def __init__(self, x, y, cout, heuristique):
        self.x = int(x)
        self.y = int(y)
        self.cout = cout
        self.heuristique = heuristique

    def memePoint(self, other):
        return self.x == other.x and self.y == other.y

    def __cmp__(self, other):
        if self.heuristique < other.heuristique:
            return 1
        elif self.heuristique == other.heuristique:
            return 0
        else:
            return -1


class FilePrio(object):
    """ A neat min-heap wrapper which allows storing items by priority
        and get the lowest item out first (pop()).
        Also implements the iterator-methods, so can be used in a for
        loop, which will loop through all items in increasing priority order.
        Remember that accessing the items like this will iteratively call
        pop(), and hence empties the heap! """

    def __init__(self):
        """ create a new min-heap. """
        self._heap = []

    def push(self, noeud):
        """ Push an item with priority into the heap.
            Priority 0 is the highest, which means that such an item will
            be popped first."""
        heapq.heappush(self._heap, (noeud.heuristique+random()/100, noeud))

    def pop(self):
        """ Returns the item with lowest priority. """
        item = heapq.heappop(self._heap)[1] # (prio, item)[1] == item
        return item

    def liste_valeurs(self):
        result = []
        for e in self._heap:
            result.append(e[1])
        return result

    def __len__(self):
        return len(self._heap)

    def __iter__(self):
        """ Get all elements ordered by asc. priority. """
        return self

    def next(self):
        """ Get all elements ordered by their priority (lowest first). """
        try:
            return self.pop()
        except IndexError:
            raise StopIteration


class Object:
    # this is a generic object: the player, a monster, an item, the stairs...
    # it's always represented by a character on screen.
    def __init__(self, x, y, char, color, blocks=False):
        self.x = int(x)
        self.y = int(y)
        self.char = char
        self.color = color
        self.blocks = blocks

    def move(self, dx, dy):
        # move by the given amount, if the destination is not blocked
        if not map[self.x + dx][self.y + dy].blocked:
            self.x += dx
            self.y += dy

    def move2(self, x, y):
        dx = x - self.x
        dy = y - self.y
        self.move(dx, dy)

    def draw(self):
        # only show if it's visible to the player
        if libtcod.map_is_in_fov(fov_map, int(self.x), int(self.y)):
            # set the color and then draw the character that represents this object at its position
            libtcod.console_set_default_foreground(con, self.color)
            libtcod.console_put_char(con, int(self.x), int(self.y), self.char, libtcod.BKGND_NONE)

    def clear(self):
        # erase the character that represents this object
        libtcod.console_put_char(con, int(self.x), int(self.y), ' ', libtcod.BKGND_NONE)

    def move_towards(self, target_x, target_y):
        # vector from this object to the target, and distance
        dx = target_x - self.x
        dy = target_y - self.y
        distance = sqrt(dx ** 2 + dy ** 2)

        # normalize it to length 1 (preserving direction), then round it and
        # convert to integer so the movement is restricted to the map grid
        dx = round(dx / distance)
        dy = round(dy / distance)
        self.move(dx, dy)

    def move_astar3(self, target):
        cases_libres, murs_vus = self.lit_frontieres()
        murs_vus = set(murs_vus)
        depart = Noeud(self.x, self.y, 0, 0)

        closedList = []
        openList = FilePrio()
        parent = dict()

        openList.push(depart)
        while len(openList) > 0:
            u = openList.pop()
            if u.x == target[0] and u.y == target[1]:
                chemin = [u]
                while u in parent:
                    u = parent[u]
                    chemin.append(u)
                for i in range(len(chemin)):
                    chemin[i] = (chemin[i].x, chemin[i].y)
                return chemin[::-1][1:]

            for k in range(-1, 2):
                for l in range(-1, 2):
                    # if k==0 or l==0: #pour l'empecher de se déplacer en diago
                    if k == 0 or l == 0:
                        cout = 1  # On se déplace en ligne droite
                    else:
                        cout = sqrt(2)  # On se déplace en diago (pour éviter que l'algo pense qu'un chemin en zigzag
                                        # équivaut à une ligne droite)
                    v = Noeud(u.x + k, u.y + l, u.cout + cout, 0)  # Si le noeud s'avère intéressant, l'heuristique
                                                                   # sera alors calculée
                    present = False
                    if (v.x, v.y) not in murs_vus:
                        for n in closedList + openList.liste_valeurs():
                            if n.memePoint(v) and n.cout <= v.cout:
                                present = True
                        if not present:
                            v.heuristique = v.cout + dist((v.x, v.y), (target[0], target[1]))
                            openList.push(v)
                            parent[v] = u
            closedList.append(u)
        print("Aucun chemin trouvé")
        return -1

    def lit_frontieres(self):
        cases_libres = []
        murs_vus = []
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j].explored and not map[i][j].blocked:
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if not map[i+k][j+l].explored:
                                    cases_libres.append((i+k, j+l))
                                    break
                elif map[i][j].explored and map[i][j].blocked:
                    murs_vus.append((i, j))
        return cases_libres, murs_vus


def create_room(room):
    global map
    # go through the tiles in the rectangle and make them passable
    for x in range(room.x1 + 1, room.x2):
        for y in range(room.y1 + 1, room.y2):
            map[x][y].blocked = False
            map[x][y].block_sight = False

    # créer piliers
    for x in range(room.x1 + 1, room.x2):
        for y in range(room.y1 + 1, room.y2):
            if randint(0, 1300-room.size) == 0:
                map[x][y].blocked = True
                map[x][y].block_sight = True


def create_h_tunnel(x1, x2, y):
    global map
    # horizontal tunnel. min() and max() are used in case x1>x2
    for x in range(int(min(x1, x2)), int(max(x1, x2) + 1)):
        map[x][int(y)].blocked = False
        map[x][int(y)].block_sight = False
        map[x][int(y)+1].blocked = False
        map[x][int(y)+1].block_sight = False
        map[x][int(y)-1].blocked = False
        map[x][int(y)-1].block_sight = False


def create_v_tunnel(y1, y2, x):
    global map
    # vertical tunnel
    for y in range(int(min(y1, y2)), int(max(y1, y2) + 1)):
        map[int(x)][y].blocked = False
        map[int(x)][y].block_sight = False
        map[int(x)+1][y].blocked = False
        map[int(x)+1][y].block_sight = False
        map[int(x)-1][y].blocked = False
        map[int(x)-1][y].block_sight = False


def make_map():
    global map, player

    # fill map with "blocked" tiles
    map = [[Tile(True)
        for y in range(MAP_HEIGHT)]
            for x in range(MAP_WIDTH)]

    rooms = []
    num_rooms = 0

    for r in range(MAX_ROOMS):
        # random width and height
        w = libtcod.random_get_int(0, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        h = libtcod.random_get_int(0, ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        # random position without going out of the boundaries of the map
        x = libtcod.random_get_int(0, 0, MAP_WIDTH - w - 1)
        y = libtcod.random_get_int(0, 0, MAP_HEIGHT - h - 1)

        # "Rect" class makes rectangles easier to work with
        new_room = Rect(x, y, w, h)

        # run through the other rooms and see if they intersect with this one
        failed = False
        for other_room in rooms:
            if new_room.intersect(other_room):
                failed = True
                break

        if not failed:
            # this means there are no intersections, so this room is valid

            # "paint" it to the map's tiles
            create_room(new_room)

            # center coordinates of new room, will be useful later
            (new_x, new_y) = new_room.center()

            if num_rooms == 0:
                # this is the first room, where the player starts at
                player.x = new_x
                player.y = new_y
            else:
                # all rooms after the first:
                # connect it to the previous room with a tunnel

                # center coordinates of previous room
                (prev_x, prev_y) = rooms[num_rooms-1].center()

                # draw a coin (random number that is either 0 or 1)
                if libtcod.random_get_int(0, 0, 1) == 1:
                    # first move horizontally, then vertically
                    create_h_tunnel(prev_x, new_x, prev_y)
                    create_v_tunnel(prev_y, new_y, new_x)
                else:
                    # first move vertically, then horizontally
                    create_v_tunnel(prev_y, new_y, prev_x)
                    create_h_tunnel(prev_x, new_x, new_y)

            # finally, append the new room to the list
            rooms.append(new_room)
            num_rooms += 1


def dist(x, y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def render_all():
    global fov_map, color_dark_wall, color_light_wall
    global color_dark_ground, color_light_ground
    global fov_recompute

    if fov_recompute:
        # recompute FOV if needed (the player moved or something)
        fov_recompute = False
        libtcod.map_compute_fov(fov_map, int(player.x), int(player.y), TORCH_RADIUS, FOV_LIGHT_WALLS, FOV_ALGO)

        # go through all tiles, and set their background color according to the FOV
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                visible = libtcod.map_is_in_fov(fov_map, x, y)
                wall = map[x][y].block_sight
                if not visible:
                    # if it's not visible right now, the player can only see it if it's explored
                    if map[x][y].explored:
                        if wall:
                            libtcod.console_set_char_background(con, x, y, color_dark_wall, libtcod.BKGND_SET)
                        else:
                            libtcod.console_set_char_background(con, x, y, color_dark_ground, libtcod.BKGND_SET)
                else:
                    # it's visible
                    if wall:
                        libtcod.console_set_char_background(con, x, y, color_light_wall, libtcod.BKGND_SET)
                    else:
                        libtcod.console_set_char_background(con, x, y, color_light_ground, libtcod.BKGND_SET)
                    # since it's visible, explore it
                    map[x][y].explored = True

    # draw all objects in the list
    for object in objects:
        object.draw()

    # blit the contents of "con" to the root console
    libtcod.console_blit(con, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, 0)


def handle_keys():
    global fov_recompute

    key = libtcod.console_check_for_keypress()  # real-time
    # key = libtcod.console_wait_for_keypress(True)  #turn-based

    if key.vk == libtcod.KEY_ENTER and key.lalt:
        # Alt+Enter: toggle fullscreen
        libtcod.console_set_fullscreen(not libtcod.console_is_fullscreen())

    elif key.vk == libtcod.KEY_ESCAPE:
        return True  # exit game

    # movement keys
    if libtcod.console_is_key_pressed(libtcod.KEY_UP):
        player.move(0, -1)
        fov_recompute = True

    elif libtcod.console_is_key_pressed(libtcod.KEY_DOWN):
        player.move(0, 1)
        fov_recompute = True

    elif libtcod.console_is_key_pressed(libtcod.KEY_LEFT):
        player.move(-1, 0)
        fov_recompute = True

    elif libtcod.console_is_key_pressed(libtcod.KEY_RIGHT):
        player.move(1, 0)
        fov_recompute = True


if __name__ == '__main__':

    #############################################
    # Initialization & Main Loop
    #############################################

    libtcod.console_set_custom_font('arial10x10.png', libtcod.FONT_TYPE_GREYSCALE | libtcod.FONT_LAYOUT_TCOD)
    libtcod.console_init_root(SCREEN_WIDTH, SCREEN_HEIGHT, 'python/libtcod tutorial', False)
    libtcod.sys_set_fps(LIMIT_FPS)
    con = libtcod.console_new(SCREEN_WIDTH, SCREEN_HEIGHT)

    # create object representing the player
    player = Object(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, '@', libtcod.white)

    # create an NPC
    npc = Object(SCREEN_WIDTH/2 - 5, SCREEN_HEIGHT/2, '@', libtcod.yellow)

    # the list of objects with those two
    objects = [npc, player]

    # generate map (at this point it's not drawn to the screen)
    make_map()

    # create the FOV map, according to the generated map
    fov_map = libtcod.map_new(MAP_WIDTH, MAP_HEIGHT)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            libtcod.map_set_properties(fov_map, x, y, not map[x][y].block_sight, not map[x][y].blocked)


    fov_recompute = True

    chemin = []
    # MLP
    n = MAP_HEIGHT * MAP_WIDTH
    network = MLP(n, 2 * n, 2 * n, n)  # en entrée la map et point de départ et d'arrivée
    # 1 pour start/end, -1 pour mur, 0 pour libre

    while not libtcod.console_is_window_closed():

        # render the screen
        render_all()

        ##### génère le set d'entrainement #####
        free, blocked = player.lit_frontieres()
        # points de départ et d'arrivée générés aléatoirement en zone libre
        start_points = []
        end_points = []
        N = 100
        for k in range(N):
            i = randint(1, MAP_WIDTH-1)  # j'ai mis 1 et max-1 à cause d'erreurs out of range
            j = randint(1, MAP_HEIGHT-1)
            while map[i][j].blocked:
                i = randint(1, MAP_WIDTH-1)
                j = randint(1, MAP_HEIGHT-1)
            start_points.append((i, j))
        for k in range(N):
            i = randint(1, MAP_WIDTH-1)
            j = randint(1, MAP_HEIGHT-1)
            while map[i][j].blocked:
                i = randint(1, MAP_WIDTH-1)
                j = randint(1, MAP_HEIGHT-1)
            end_points.append((i, j))
        # Objet à créer pour les points de départ pour lancer move_astar3 (peut s'enlever en mettant move astar
        # hors classe)
        start_objects = [Object(start[0], start[1], 'o', libtcod.white) for start in start_points]
        path_list = [start_objects[i].move_astar3(end_points[i]) for i in range(N)]
        print(start_points)
        print(end_points)
        print(path_list)
        sleep(1)

        # Mise au bon format pour le réseau
        samples = np.zeros(N, dtype=[('input', float, n), ('output', float, n)])
        init = []
        result = []
        for k in range(N):
            init.append(map_to_input(map, start_points[k], end_points[k]))
            result.append(map_to_result(map, start_points[k], end_points[k], path_list[k]))
            samples[k] = init[k], result[k]
        print("En cours d'entrainement")
        learn_path_format2(network, samples, epochs=1000, lrate=0.1, momentum=0.2)

        print("Entrainement terminé")

        break  # Pour ne pas boucler

        fov_recompute = True

        libtcod.console_flush()

        # erase all objects at their old locations, before they move
        for object in objects:
            object.clear()

        # handle keys and exit game if needed
        exit = handle_keys()
        if exit:
            break
