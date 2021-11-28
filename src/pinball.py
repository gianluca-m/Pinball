import numpy as np
import math
import tkinter as tk
import time

'''
    adjusted from: https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/04-pinball.html
'''
    
cHeight = 900        
cWidth = int(3 / 5 * cHeight)

window = tk.Tk()
window.wm_title("Pinball")
canvas = tk.Canvas(window, bg="black", width=cWidth, height=cHeight)
canvas.pack()


# utility functions
def vector_length(x):
    '''Returns the length of a vector'''
    return np.sqrt(x.dot(x))    # faster than np.linalg.norm(x)


def closest_point_on_segment(p, a, b):
    '''Returns the closest point from point 'p' on the segment 'a' to 'b' '''
    ab = b - a
    t = ab.dot(ab)

    # a = b -> closest point is a resp. b
    if (t == 0.0):
        return np.copy(a)

    t = max(0.0, min(1.0, (p.dot(ab) - a.dot(ab)) / t))
    return a + ab * t


def angle_between_vectors(a, b):
    '''Returns angle between vectors 'a' and 'b' in Rad'''
    unit_a = a / vector_length(a)
    unit_b = b / vector_length(b)
    return np.arccos(np.clip(unit_a.dot(unit_b), -1.0, 1.0))


class Ball:
    def __init__(self, pos, vel, radius, mass, restitution):
        self.radius = radius
        self.mass = mass
        self.pos = pos
        self.vel = vel 
        self.restitution = restitution


    def simulate(self, dt, g):
        self.vel += g * dt
        self.pos += self.vel * dt

    

class Flipper:
    def __init__(self, pos, length, radius, rest_angle, max_rotation, angular_vel, window, button):
        self.pos = pos
        self.length = length
        self.radius = radius
        self.rest_angle = rest_angle
        self.max_rotation = abs(max_rotation)
        self.sign = math.copysign(1, max_rotation)
        self.angular_vel = angular_vel
        
        self.rotation = 0.0
        self.current_angular_vel = 0.0
        self.is_pressed = False

        window.bind(f'<KeyPress-{button}>', self.activate)
        window.bind(f'<KeyRelease-{button}>', self.deactivate)

    
    def simulate(self, dt):
        prev_rotation = self.rotation
        
        if (self.is_pressed):
            self.rotation = min(self.rotation + dt * self.angular_vel, self.max_rotation)
        else:
            self.rotation = max(self.rotation - dt * self.angular_vel, 0.0)
        
        self.current_angular_vel = self.sign * (self.rotation - prev_rotation) / dt

    def select(self, pos):
        dist = self.pos - pos
        return vector_length(dist) - vector_length(self.length)


    def activate(self, event):
        self.is_pressed = True

    
    def deactivate(self, event):
        self.is_pressed = False


    def rotate(self, angle):
        # Rotation: Translate Middle to the Origin, do simple rotation, translate back
        # Clockwise 45 degrees = r = np.array(((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta))))
        theta = np.radians(angle)
        trans_x = self.pos[5][0]
        trans_y = self.pos[5][1]

        c = np.copy(self.pos)
        # rotation
        for i in c:
            # NP Array of 2D Vect
            # Translate to Origin
            i[0] = i[0] - trans_x
            i[1] = i[1] - trans_y
            tmpx = i[0]
            tmpy = i[1]

            i[0] = np.cos(theta) * tmpx + (-np.sin(theta)) * tmpy
            i[1] = np.sin(theta) * tmpx + np.cos(theta) * tmpy
                
            i[0] = i[0] + trans_x
            i[1] = i[1] + trans_y
        return c



class CircleObstacle:
    def __init__(self, pos, radius, push_vel):
        self.pos = np.copy(pos)
        self.radius = radius
        self.push_vel = push_vel



class Shooter:
    def __init__(self, pos, rest_pos, k, mass, button):
        self.pos = np.copy(pos)
        self.rest_pos = np.copy(rest_pos)
        self.k = k
        self.mass = mass
        self.push_vel = 0.0
        self.is_pressed = False

        window.bind(f'<KeyPress-{button}>', self.activate)
        window.bind(f'<KeyRelease-{button}>', self.deactivate)


    def simulate(self, dt):
        if (self.is_pressed):
            self.pos[0][1] = max(self.pos[0][1] + 1, self.rest_pos[1])
            # F = -kx
            force = -self.k * (self.rest_pos[1] - self.pos[0][1])
            # F = m*a --> a = F/m
            a = force / self.mass
            self.push_vel = a * dt
        else:
            self.pos[0][1] = self.rest_pos[1]


    def activate(self, event):
        self.is_pressed = True


    def deactivate(self, event):
        self.is_pressed = False



class PinballScene:
    def __init__(self, border, balls, obstacles, shooters, flippers, g=np.array([0, 981]), dt=1/60):
        self.border = border
        self.balls = balls
        self.obstacles = obstacles
        self.shooters = shooters
        self.flippers = flippers
        self.g = g
        self.dt = dt
        
        self.score = 0
        self.paused = True


    @staticmethod
    def handle_ball_ball_collision(ball1: Ball, ball2: Ball):
        restitution = min(ball1.restitution, ball2.restitution)
        dir = ball2.pos - ball1.pos
        dist = vector_length(dir)
        if (dist == 0.0 or dist > ball1.radius + ball2.radius):
            # no collision
            return

        dir *= (1.0 / dist)     # normalize

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


    def handle_ball_circle_obstacle_collision(self, ball: Ball, obstacle: CircleObstacle):
        dir = ball.pos - obstacle.pos
        dist = vector_length(dir)
        if (dist == 0.0 or dist > ball.radius + obstacle.radius):
            # no collision
            return

        dir *= (1.0 / dist)     # normalize

        corr = ball.radius + obstacle.radius - dist
        ball.pos += dir * corr

        vel = ball.vel.dot(dir)
        ball.vel += dir * (obstacle.push_vel - vel)

        # update score
        self.score += 1


    @staticmethod
    def handle_ball_shooter_collision(ball: Ball, shooter: Shooter):
        if (ball.pos[1] + ball.radius < shooter.pos[0][1]):
            # No Collision
            return

        ball.pos[1] = shooter.pos[0][1] - ball.radius

        if(shooter.is_pressed):
            # We are charging the Shooter so we dont launch (just bounce)
            ball.vel[1] = 0.95 * (-ball.vel[1])
            return

        # Update Velocity in  normal state
        ball.vel[1] = 0.6 * (-ball.vel[1])  - (shooter.push_vel)


    @staticmethod
    def handle_ball_flipper_collision(ball: Ball, flipper: Flipper):
        rotated = flipper.rotate(flipper.rest_angle + flipper.rotation * flipper.sign)
        closest = closest_point_on_segment(ball.pos, rotated[5], rotated[2])

        dir = ball.pos - closest
        dist = vector_length(dir)
        if (dist == 0.0 or dist > ball.radius + flipper.radius):
            # no collision
            return

        dir *= (1.0 / dist)     # normalize
        corr = ball.radius + flipper.radius - dist
        ball.pos += dir * corr

        # update velocity
        radius = np.copy(closest)
        radius += (dir * flipper.radius)
        radius -= flipper.pos[5]
        surface_vel = np.array([-radius[1], radius[0]])
        surface_vel *= flipper.current_angular_vel / 35

        v = ball.vel.dot(dir)
        new_v = surface_vel.dot(dir)
        ball.vel += dir * (new_v - v)


    @staticmethod
    def handle_ball_border_collision(ball: Ball, border):
        # find closest segment
        min_dist = 0.0
        closest = np.array([])
        normal = np.array([1.0, 0.0])

        for i in range(len(border) - 1):
            a = border[i]
            b = border[(i+1)]
            c = closest_point_on_segment(ball.pos, a, b)
            dir = ball.pos - c
            dist = vector_length(dir)

            ab = a - b
            curr_normal = np.array([-ab[1], ab[0]])

            # if closest point 'c' is one of the border endpoints, the dist can be equal for two borders, thus need another way to determine closest border:
            #   use angle between dir = (ball.pos - c) and either the current inverted normal or the invereted normal of the currently closest border
            angle_dir_curr_normal = min(angle_between_vectors(dir, curr_normal), angle_between_vectors(curr_normal, dir))
            angle_dir_min_normal = min(angle_between_vectors(dir, normal), angle_between_vectors(normal, dir))

            if (i == 0 or dist < min_dist or (dist == min_dist and angle_dir_curr_normal < angle_dir_min_normal)):
                min_dist = dist
                closest = c
                ab = a - b
                normal = curr_normal        # This is the left-normal
                                            # We need it because we build the Polygonal
                                            # From Counter-Clockwise
                                            # This is the normal "Inside" the Pinball game

        dir = ball.pos - closest
        dist = vector_length(dir)

        unit_dir = dir / dist

        if (dir.dot(normal) >= 0.0) and (dist > ball.radius):     
            # if on correct side of border (i.e. inside canvas) and distance from closest point on border to ball is smaller than radius
            return  # no collision
        
        if dir.dot(normal) < 0.0:
            # on wrong side of border, thus move ball in opposite direction so that ball is inside again
            unit_dir *= -1.0

        # dist < ball.radius or ball is on wrong side of border
        # move ball so whole ball inside of canvas
        ball.pos = closest + unit_dir * ball.radius
        
        # update velocity
        ball.vel -= (2.0 - ball.restitution) * (ball.vel.dot(unit_dir)) * unit_dir  # https://math.stackexchange.com/a/13266


    def simulate(self):    
        self.flippers[0].simulate(self.dt)
        self.flippers[1].simulate(self.dt)
        self.shooters[0].simulate(self.dt)

        for ball in self.balls:
            ball.simulate(self.dt, self.g)

        for i in range(len(self.balls)):
            ball = self.balls[i]
            
            # BROAD PHASE COLLISION DETECTION
            if (ball.pos[0] > cWidth / 2):
                if (ball.pos[1] > cHeight / 2):
                    # BOTTOM RIGHT
                    self.handle_ball_circle_obstacle_collision(ball, self.obstacles[2])
                    self.handle_ball_flipper_collision(ball, self.flippers[1])
                    self.handle_ball_border_collision(ball, self.border[3:10])
                    # Ball-Shooter Effect 
                    if (ball.pos[0] >= cWidth -50):
                        self.handle_ball_shooter_collision(ball, self.shooters[0])
                else: 
                    # TOP RIGHT
                    self.handle_ball_circle_obstacle_collision(ball, self.obstacles[1])
                    self.handle_ball_border_collision(ball, self.border[[7, 8, 9, 10, 0]])
            else:
                if (ball.pos[1] > cHeight / 2):
                    # BOTTOM LEFT
                    self.handle_ball_circle_obstacle_collision(ball, self.obstacles[3])
                    self.handle_ball_flipper_collision(ball, self.flippers[0])
                    self.handle_ball_border_collision(ball, self.border[:5])
                else:
                    # TOP LEFT
                    self.handle_ball_circle_obstacle_collision(ball, self.obstacles[0])
                    self.handle_ball_border_collision(ball, self.border[[10, 0, 1]])

            for j in range(i+1, len(self.balls)):
                self.handle_ball_ball_collision(ball, self.balls[j])    



def setup_scene() -> PinballScene:
    global window
    # scene borders --> Define set of pixel pairs
    border = np.array([[0.0, 0.0], [0.0,cHeight*0.75], [cWidth*0.3,cHeight*0.9], [cWidth*0.3,cHeight], [cWidth*0.7,cHeight], [cWidth*0.7,cHeight*0.9], [cWidth-40, cHeight*0.75], [cWidth-40,cHeight], [cWidth,cHeight], [cWidth,60], [cWidth-60,0.0]])

    # balls
    radius = 10
    mass = math.pi * radius**2
    restitution = 0.2
    pos1 = np.array([cWidth * 0.25, cHeight * 0.05])
    vel1 = np.array([-1500.0, 0.0])
    ball1 = Ball(pos1, vel1, radius, mass, restitution)

    pos2 = np.array([cWidth * 0.01, cHeight * 0.8])
    vel2 = np.array([0.0, 0.0])
    ball2 = Ball(pos2, vel2, radius, mass, restitution)

    pos3 = np.array([cWidth - 40, cHeight * 0.45])
    vel3 = np.array([0.0, 0.0])
    ball3 = Ball(pos3, vel3, radius, mass, restitution)
    balls = [ball1, ball2, ball3]

    # obstacles
    obstacles = []
    obstacles.append(CircleObstacle(np.array([0.25 * cWidth, 0.2 * cHeight]), 70, 200.0))
    obstacles.append(CircleObstacle(np.array([0.75 * cWidth, 0.4 * cHeight]), 50, 500.0))
    obstacles.append(CircleObstacle(np.array([0.65 * cWidth, 0.7 * cHeight]), 40, 300.0))
    obstacles.append(CircleObstacle(np.array([0.2 * cWidth, 0.61 * cHeight]), 50, 1000.0))

    # shooters
    shooters = []
    fixed_pos = np.array([[cWidth - 40, cHeight - 60], [cWidth, cHeight]])
    shooters.append(Shooter(fixed_pos, np.copy(fixed_pos[0]), 2000, 1, 'k'))

    # flippers
    radius = int(cWidth * 0.02)
    length = int(cWidth * 0.15)
    max_rotation = -80
    rest_angle = 30
    angular_vel = 1000
    restitution = 1.0
    x1 = cWidth * 0.3 
    y1 = cHeight * 0.9 
    x2 = cWidth * 0.7 
    y2 = cHeight * 0.9
    flipper1 = Flipper(np.array([[x1,y1+radius+radius], [x1+length,y1+radius+radius], [x1+length,y1+radius], [x1+length,y1], [x1,y1], [x1,y1+radius]]), length, radius, rest_angle, max_rotation, angular_vel, window, 'a')
    flipper2 = Flipper(np.array([[x2,y2+radius+radius], [x2-length,y2+radius+radius], [x2-length,y2+radius], [x2-length,y2], [x2,y2], [x2,y2+radius]]), length, radius, -rest_angle, -max_rotation, angular_vel, window, 'd')
    flippers = [flipper1, flipper2]

    pinball_scene = PinballScene(border, balls, obstacles, shooters, flippers)
    return pinball_scene


def draw_disc(x, y, radius, col):
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=col, outline='')


def draw(pinball_scene: PinballScene):
    global canvas
    # if we don't delete all canvas objects, tkinter will keep all objects in memory and always create new ones when drawing another object
    # this will create performance issues and potentially out of memory issue
    # TODO: maybe use "move" method for certain objects instead of redrawing (?)
    canvas.delete("all")
    
    # Draw Frame around GUI:
    canvas.create_polygon(pinball_scene.border.flatten().tolist(), outline="black", fill="white")

    # Draw the balls:
    for b in pinball_scene.balls:
        draw_disc(b.pos[0], b.pos[1], b.radius, "green")

    # Draw the obstacles:
    for o in pinball_scene.obstacles:
        draw_disc(o.pos[0], o.pos[1], o.radius, "blue")

    # Draw the Shooters:
    for s in pinball_scene.shooters:
        canvas.create_rectangle(s.pos.flatten().tolist(), fill='red')
  
    # Draw the flippers
    for f in pinball_scene.flippers:
        new_coords = f.rotate(f.rest_angle + f.rotation * f.sign)
        coords = new_coords.flatten().tolist()
        canvas.create_polygon(coords, fill = "red")
        draw_disc(coords[4], coords[5], f.radius, "red")
        draw_disc(coords[10], coords[11], f.radius, "red")

    # Draw the score
    canvas.create_text(cWidth * 0.5, cHeight * 0.1, text=f'Score: {pinball_scene.score}', fill='red', font=('arial bold', 18))

    canvas.update()


def update(pinball_scene: PinballScene):
    while True:
        pinball_scene.simulate()
        draw(pinball_scene)
        time.sleep(pinball_scene.dt)


def main():
    pinball_scene = setup_scene()    
    draw(pinball_scene)


start_button = tk.Button(window, text='START', font=('arial bold', 18), height=2, width=10,
    bg="black", fg="white", activebackground="green", relief="raised", command=lambda:(update(setup_scene())))
start_button.pack(side=tk.BOTTOM, anchor=tk.S)

# TODO: this doesn't quite work yet. It still shows exception messages when closing the window
window.protocol("WM_DELETE_WINDOW", window.destroy)     # properly close tkinter window when pressing X button, so we don't get all the exception messages

if __name__ == "__main__":
    main()

window.mainloop()
