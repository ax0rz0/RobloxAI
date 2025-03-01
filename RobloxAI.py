import time
import cv2
import mss
import numpy as np
import torch
import json
from ultralytics import YOLO
from pynput.keyboard import Controller, Key

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

MOVEMENT_KEYS = {
    'forward': 'w',
    'left': 'a',
    'right': 'd',
    'backward': 's',
    'look_left': Key.left,
    'look_right': Key.right,
    'jump': Key.space
}

keyboard = Controller()

LEARNED_OBJECTS_FILE = "learned_objects.json"

def get_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Use the first screen
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Fix RGB error
        return img

class Player:
    def __init__(self):
        self.x, self.y = 0, 0
        self.stuck_counter = 0
        self.learned_objects = self.load_learned_objects()
        self.rest_timer = 0
        self.curiosity_active = False

    def load_learned_objects(self):
        try:
            with open(LEARNED_OBJECTS_FILE, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_learned_objects(self):
        with open(LEARNED_OBJECTS_FILE, 'w') as file:
            json.dump(self.learned_objects, file)

    def press_keys(self, keys, duration=0.1):
        for key in keys:
            keyboard.press(key)
        time.sleep(duration)
        for key in keys:
            keyboard.release(key)

    def move(self, direction, duration=3.0):
        key = MOVEMENT_KEYS.get(direction)
        if key:
            print(f"[AI] Moving {direction}")
            self.press_keys([key], duration)
            self.x += {'left': -1, 'right': 1}.get(direction, 0)
            self.y += {'forward': 1, 'backward': -1}.get(direction, 0)
            self.stuck_counter += 1

    def jump(self):
        print("[AI] Jumping")
        self.press_keys([MOVEMENT_KEYS['jump']], 0.1)
        self.stuck_counter = 0

    def look_around(self):
        angle = np.random.uniform(0.2, 2.0)
        if np.random.rand() > 0.5:
            print("[AI] Looking left")
            self.press_keys([MOVEMENT_KEYS['look_left']], angle)
        else:
            print("[AI] Looking right")
            self.press_keys([MOVEMENT_KEYS['look_right']], angle)

    def wander(self):
        if self.rest_timer > 0:
            print("[AI] Resting")
            self.rest_timer -= 1
            return

        direction = np.random.choice(list(MOVEMENT_KEYS.keys())[:4])
        wander_time = np.random.uniform(4.0, 12.0)
        self.move(direction, duration=wander_time)
        self.look_around()

        if np.random.rand() > 0.7:
            print("[AI] Taking a short rest")
            self.rest_timer = np.random.randint(5, 15)

        if np.random.rand() > 0.5 and self.learned_objects:
            entity = np.random.choice(list(self.learned_objects.keys()))
            if entity in ['person', 'dog', 'cat', 'bird']:
                print(f"[AI] Approaching detected {entity}...")
                self.curiosity_active = True
                self.move('forward', duration=np.random.uniform(3.0, 6.0))

        if self.stuck_counter > 3:
            self.jump()

    def avoid_obstacles(self, results):
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    width = x2 - x1
                    if width > 100:
                        print("[AI] Obstacle detected! Turning left...")
                        self.press_keys([MOVEMENT_KEYS['look_left']], np.random.uniform(0.3, 1.0))
                        return True
        return False

    def learn_objects(self, results):
        for result in results:
            if hasattr(result, 'names') and hasattr(result, 'boxes') and result.boxes is not None:
                for i, box in enumerate(result.boxes.xyxy):
                    name = result.names[result.boxes.cls[i].item()]
                    if name not in self.learned_objects:
                        self.learned_objects[name] = 1
                    else:
                        self.learned_objects[name] += 1
                    print(f"[AI] Learned about {name}")
        self.save_learned_objects()

player = Player()

print("[AI] Starting Roblox AI...")

while True:
    screen = get_screen()
    results = model.predict(screen, conf=0.5)

    player.learn_objects(results)

    if not player.avoid_obstacles(results):
        player.wander()

    time.sleep(0.05)
