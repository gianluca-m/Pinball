import numpy as np
import math
import tkinter as tk

from numpy.lib.function_base import flip

import time


'''
    This is basically a 1:1 copy of the code from: https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/04-pinball.html
    TODO: rewrite in own code

    TODO: add GUI stuff
'''
cWidth=400
cHeight=600
flipperHeight = 1.7
cScale = cHeight / flipperHeight
simWidth = cWidth / cScale
simHeight = cHeight / cScale
window = tk.Tk()
c = tk.Canvas(window,bg="black",width=cWidth,height=cHeight)
c.pack()

left_flip = False
right_flip = False



# utility function
def vector_length(x):
    '''returns the length of a vector'''
    return np.sqrt(x.dot(x))    # faster than np.linalg.norm(x)

#This Function contains Errors
def closest_point_on_segment(p, a, b):
    ab = b - a
    t = ab.dot(ab)

    # a = b -> closest point is a resp. b
    if (t == 0.0):
        return np.copy(a)

    t = max(0.0, min(1.0, (p.dot(ab) - a.dot(ab)) / t))
    return a + ab * t


class Ball:
    def __init__(self, pos, vel, radius, mass, restitution):
        self.radius = radius
        self.mass = mass
        self.pos = pos
        self.vel = vel 
        self.restitution = restitution


    def simulate(self, dt, g):
        self.vel[1] += g * dt
        self.pos += self.vel * dt

    

class Flipper:
    def __init__(self, pos, length, radius, rest_angle, max_rotation, angular_vel):
        self.pos = pos
        self.length = length
        self.radius = radius
        self.rest_angle = rest_angle
        self.max_rotation = abs(max_rotation)
        self.sign = math.copysign(1, max_rotation)
        self.angular_vel = angular_vel
        #dynamic
        self.rotation = 0.0
        self.current_angular_vel = 0.0
        self.touch_id = -1

    
    def simulate(self, dt, pressed):
        #global left_flip
        prev_rotation = self.rotation
        #pressed = left_flip
        
        if (pressed):
            self.rotation = 40 + min(self.rotation + dt * self.angular_vel, self.max_rotation-40)
            #left_flip = False
        else:
            self.rotation = max(self.rotation - dt * self.angular_vel, 0.0)
        
        self.current_angular_vel = self.sign * (self.rotation - prev_rotation) / dt


    def select(self, pos):
        dist = self.pos - pos
        return vector_length(dist) - vector_length(self.length)



class CircleObstacle:
    def __init__(self, pos, radius, push_vel):
        self.pos = np.copy(pos)
        self.radius = radius
        self.push_vel = push_vel



class PhysicsScene:
    def __init__(self, border, balls, obstacles, flippers, g=-9.81, dt=1/45):
        self.border = border
        self.balls = balls
        self.obstacles = obstacles
        self.flippers = flippers
        self.g = g
        self.dt = dt
        
        self.score = 0
        self.paused = True

def cX(xpos):
	return xpos * cScale

def cY(ypos):
	return cHeight - ypos * cScale


def setup_canvas():
    # TODO: gui setup
    
     #Function that runs the window + event listener
    pass

def flipper_angle(f,angle):
    #Rotation: Translate Middle to the Origin, do simple rotation, translate back
    #Clockwise 45 degrees = r = np.array(((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta))))
    theta = np.radians(angle)
    r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    trans_x = f.pos[5][0]
    trans_y = f.pos[5][1]

    c = np.copy(f.pos)
    # rotation
    for i in c:
        #NP Array of 2D Vect
        #Translate to Origin
        i[0] = i[0] - trans_x
        i[1] = i[1] - trans_y
        tmpx = i[0]
        tmpy = i[1]

        i[0] = np.cos(theta) * tmpx + (-np.sin(theta)) * tmpy
        i[1] = np.sin(theta) * tmpx + np.cos(theta) * tmpy
            
        i[0] = i[0] + trans_x
        i[1] = i[1] + trans_y
    return c
def setup_scene() -> PhysicsScene:
    #scene borders --> Define set of pixel pairs
    #border = [0.0,0.0, 0.0,800, 300,1100, 300,1200, 500,1200, 500,1100, 800,800, 800,0.0]
    #border = [x/2 for x in border]
    border = np.array([[0.0, 0.0], [cWidth, 0.0], [cWidth, cHeight], [0.0, cHeight]])

    # balls
    radius = 0.03
    mass = math.pi * radius**2
    restitution = 1
    pos1 = np.array([1, 0.5])
    vel1 = np.array([-1.2, 3.5])
    ball1 = Ball(pos1, vel1, radius, mass,restitution)

    pos2 = np.array([0.5, 0.5])
    vel2 = np.array([1.2, 3.5])
    ball2 = Ball(pos2, vel2, radius, mass,restitution)
    balls = [ball1, ball2]

    # obstacles
    obstacles = []
    obstacles.append(CircleObstacle(np.array([0.25, 0.6]), 0.1, 2.0))
    obstacles.append(CircleObstacle(np.array([0.75, 0.5]), 0.1, 2.0))
    obstacles.append(CircleObstacle(np.array([0.7, 1.0]), 0.12, 2.0))
    obstacles.append(CircleObstacle(np.array([0.2, 1.2]), 0.1, 2.0))

    # flippers
    radius = 12
    length = 50
    max_rotation = -80
    rest_angle = 40
    angular_vel = 200
    restitution = 1.0
    x1 = cWidth * 0.2 # 300
    y1 = cHeight * 0.9 # 1100
    x2 = cWidth * 0.8 # 500
    y2 = cHeight * 0.9 # 1100
    flipper1 = Flipper(np.array([[x1,y1+radius+radius], [x1+length,y1+radius+radius], [x1+length,y1+radius], [x1+length,y1], [x1,y1], [x1,y1+radius]]), length, radius, rest_angle, max_rotation, angular_vel)
    flipper2 = Flipper(np.array([[x2,y2+radius+radius], [x2-length,y2+radius+radius], [x2-length,y2+radius], [x2-length,y2], [x2,y2], [x2,y2+radius]]), length, radius, -rest_angle, -max_rotation, angular_vel)
    flippers = [flipper1, flipper2]

    physics_scene = PhysicsScene(border, balls, obstacles, flippers)
    return physics_scene


def draw_disc(x, y, radius, col):
    return c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=col,outline='')

def draw(physics_scene):
    # TODO: add
    #Draw Frame around GUI:
    #c.create_rectangle(0,0,cWidth,cHeight,outline="green")
    c.create_polygon(physics_scene.border.flatten().tolist(),outline="black",fill="white")


    #Draw the balls:
    for b in physics_scene.balls:
        draw_disc(cX(b.pos[0]),cY(b.pos[1]),b.radius*cScale,"black")


    #Draw the obstacles:
    for o in physics_scene.obstacles:
        draw_disc(cX(o.pos[0]),cY(o.pos[1]),o.radius*cScale,"blue")
  
    #Draw the flippers
    for f in physics_scene.flippers:
        print(f.rest_angle + f.rotation*f.sign)
        new_coords = flipper_angle(f,f.rest_angle + f.rotation * f.sign)
        coords = new_coords.flatten().tolist()
        c.create_polygon(coords, fill = "red")
        draw_disc(coords[4], coords[5], f.radius,"red")
        draw_disc(coords[10], coords[11],f.radius,"red")

    c.update()


def handle_ball_ball_collision(ball1: Ball, ball2: Ball):
    restitution = min(ball1.restitution, ball2.restitution)
    dir = ball2.pos - ball1.pos
    dist = vector_length(dir)
    if (dist == 0.0 or dist > ball1.radius + ball2.radius):
        # no collision
        return

    dir *= (1.0 / dist)     # normalize (?)

    corr = (ball1.radius + ball2.radius - dist) / 2.0
    ball1.pos += dir * -corr
    ball2.pos += dir * corr

    v1 = ball1.vel.dot(dir)
    v2 = ball2.vel.dot(dir)

    m1 = ball1.mass
    m2 = ball2.mass

    new_v1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    new_v2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

    ball1.vel += dir * (new_v1 - v1)
    ball2.vel += dir * (new_v2 - v2)


def handle_ball_circle_obstacle_collision(ball: Ball, obstacle: CircleObstacle):
    dir = ball.pos - obstacle.pos
    dist = vector_length(dir)
    if (dist == 0.0 or dist > ball.radius + obstacle.radius):
        # no collision
        return

    dir *= (1.0 / dist)     # normalize (?)

    corr = ball.radius + obstacle.radius - dist
    ball.pos += dir * corr

    vel = ball.vel.dot(dir)
    ball.vel += dir * (obstacle.push_vel - vel)


def handle_ball_flipper_collision(ball: Ball, flipper: Flipper):
    #Bug Call
    closest = closest_point_on_segment(ball.pos, flipper.pos[5], flipper.pos[2])
    
    dir = ball.pos - closest
    dist = vector_length(dir)
    if (dist == 0.0 or dist > ball.radius + flipper.radius):
        # no collision
        #print("No-Collision")
        return
    
    #print("Collision with Flipper!")

    dir *= (1.0 / dist)     # normalize (?)
    corr = ball.radius + flipper.radius - dist
    ball.pos += dir * corr

    # update velocity
    radius = np.copy(closest)
    radius += (dir * flipper.radius)
    radius -= flipper.pos
    surface_vel = np.array([-radius[1], radius[0]])
    surface_vel *= flipper.current_angular_vel

    v = ball.vel.dot(dir)
    new_v = surface_vel.dot(dir)
    ball.vel += dir * (new_v - v)


def handle_ball_border_collision(ball: Ball, border):
    len_border = len(border)
    if (len_border < 3):
        return

    # find closest segment
    min_dist = 0.0
    closest = np.array([])
    normal = np.array([])

    for i in range(len_border):
        a = border[i]
        b = border[(i+1) % len_border]
        c = closest_point_on_segment(ball.pos, a, b)
        d = ball.pos - c
        dist = vector_length(d)
        if (i == 0 or dist < min_dist):
            min_dist = dist
            closest = c
            ab = b - a
            normal = np.array([-ab[1], ab[0]])

    # push out
    d = ball.pos - closest
    dist = vector_length(d)
    if (dist == 0.0):
        d = normal
        dist = vector_length(normal)
    d *= (1.0 / dist)       # normalize (?)

    if (d.dot(normal) >= 0.0):
        if (dist > ball.radius):
            # no collision
            return
        ball.pos += d * (ball.radius - dist)
    else:
        ball.pos += d * -(dist + ball.radius)

    # update velocity
    v = ball.vel.dot(d)
    new_v = abs(v) * ball.restitution
    ball.vel += d * (new_v - v)

def set_left_flip(event):
    global left_flip
    left_flip = True

def set_right_flip(event):
    global right_flip
    right_flip = True

def simulate(physics_scene: PhysicsScene):
    draw(physics_scene)
    global left_flip
    global right_flip
    
    physics_scene.flippers[0].simulate(physics_scene.dt, left_flip)
    physics_scene.flippers[1].simulate(physics_scene.dt, right_flip)

    for i in range(len(physics_scene.balls)):
        ball = physics_scene.balls[i]

        # if more than 2 balls, this needs to be done differently
        #   for j = 0, j < len(balls), j++
        #       if j != i then handle_collision
        for j in range(i+1, len(physics_scene.balls)):
            handle_ball_ball_collision(ball, physics_scene.balls[j])

        for j in range(len(physics_scene.obstacles)):
            handle_ball_circle_obstacle_collision(ball, physics_scene.obstacles[j])
        
        for j in range(len(physics_scene.flippers)):
            handle_ball_flipper_collision(ball, physics_scene.flippers[j])

        handle_ball_border_collision(ball, physics_scene.border)

    for ball in physics_scene.balls:
        ball.simulate(physics_scene.dt, physics_scene.g)

    
def update(physics_scene: PhysicsScene):
    global left_flip
    global right_flip
    draw(physics_scene)
    while True:
        simulate(physics_scene)
        left_flip = False
        right_flip = False
        time.sleep(physics_scene.dt)
        
def main():
    setup_canvas()
    physics_scene = setup_scene()
    #update(physics_scene)
    draw(physics_scene)

window.bind('<KeyPress-a>', set_left_flip)
window.bind('<KeyPress-d>', set_right_flip)

start_button = tk.Button(window, text='START', font=('arial bold', 18), height=10, width=10,
    bg="black", fg="white", activebackground="green", relief="raised", command=lambda:update(setup_scene()))
start_button.pack(side=tk.BOTTOM, anchor=tk.S)

if __name__ == "__main__":
    main()

window.mainloop()

    



