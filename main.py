import math
from typing import Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sphere:
    def __init__(self, 
                 center : list[float], 
                 radius : float, 
                 color  : list[float],
                 ) -> None:
        # Sphere's spatial information.
        self.center     = torch.tensor(center, dtype=torch.float32).to(device)
        self.radius     = radius

        # Sphere's color information.
        # Let's define ambient, diffuse and specular simply.
        self.color      = torch.tensor(color, dtype=torch.float32).to(device)
        self.ambient    = torch.tensor([0.4] * 3, dtype=torch.float32).to(device)
        self.diffuse    = torch.tensor([0.5] * 3, dtype=torch.float32).to(device)
        self.specular   = torch.tensor([0.8] * 3, dtype=torch.float32).to(device)
        self.shininess  = 16


class RayTracingModule(nn.Module):

    def __init__(self,
                 image_width    : int, 
                 image_height   : int,
                 ) -> None:
        super(RayTracingModule, self).__init__()

        # Initialize rendering configuration(e.g. width, height, viewport shape etc).
        self.image_width = image_width
        self.image_height = image_height
        self.aspect_ratio = image_width / image_height
        # [WARNING!] This is so hacky code. In the viewport's uv coordinates, we set the top-left to (-1, ., 1) 
        # and the bottom-right to (1, ., -1) by arranging the order of top and bottom differently
        # from the left and right.
        self.viewport = (
                            -1.0,                       # left
                            1.0,                        # right
                            1.0 / self.aspect_ratio,    # top
                            -1.0 / self.aspect_ratio    # bottom
                         )

        # Output image
        self.image = None

        # self.gradient is used for drawing background.
        v_steps = torch.linspace(-1.0, 1.0, image_height, dtype=torch.float32).to(device)
        top_color = torch.tensor([0.5, 0.7, 1.0], dtype=torch.float32).to(device)
        bottom_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        self.gradient = torch.clamp((1 - v_steps[:, None]) * top_color + v_steps[:, None] * bottom_color, min=0.0, max=1.0)

        # [WARNING] Current code simple fixes the camera position to (0.0, 0.0, 0.0).
        # This simplifies the code since it removes the need to consider the camera's rotation axis(roll, pitch, and yaw).
        self.camera = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(device)

        # [WARNING] Since we fixed the camera position, we can initialize self.pixel_positions in advance.
        # self.pixel_positions is used for obtaining each pixel's positon.
        # We will set the self.pixel_positions y value 5 steps away along the y-axis from the camera position.
        u_steps = torch.linspace(self.viewport[0], self.viewport[1], image_width, dtype=torch.float32).to(device)
        v_steps = torch.linspace(self.viewport[2], self.viewport[3], image_height, dtype=torch.float32).to(device)
        u_steps_grid, v_steps_grid = torch.meshgrid(u_steps, v_steps, indexing='xy')
        self.pixel_positions = torch.stack((u_steps_grid, torch.full_like(u_steps_grid, 5.0), v_steps_grid), dim=-1).reshape(-1, 3)
        
        # [WARNING] Since we fixed the camera position, we can initialize self.ray_directions and self.ray_origins in advance.
        self.ray_directions = nn.functional.normalize(self.pixel_positions - self.camera, dim=-1)
        self.ray_origins = self.camera.expand(self.ray_directions.shape[0], -1)

        
    def forward(self, 
                scene : list[Sphere], 
                light_pos : torch.Tensor, 
                light_color : torch.Tensor,
                ) -> torch.Tensor:
        # Fist, use self.gradient to draw the background on self.image.
        self.image = self.gradient.unsqueeze(1).repeat(1, self.image_width, 1).reshape(-1, 3)

        # Then, calculate the hit points between the ray and sphere.
        # If the ray hits more than one sphere, select the shortest hit distance case.
        sphere_centers = torch.stack([sphere.center for sphere in scene])
        sphere_radiuses = torch.tensor([sphere.radius for sphere in scene], dtype=torch.float32).to(device)
        min_t, min_idx, valid_hit = self.ray_hit_sphere(self.ray_origins, self.ray_directions, sphere_centers, sphere_radiuses)
        
        hit_points = self.ray_origins + min_t[:, None] * self.ray_directions
        hit_sphere_centers = torch.index_select(sphere_centers, 0, min_idx)
        hit_normals = nn.functional.normalize(hit_points - hit_sphere_centers, dim=-1)

        for i, sphere in enumerate(scene):
            sphere_i_hit = valid_hit & (min_idx == i)
            
            # If a ray hits a sphere, execute the following conditional statement.
            # In other words, only calculate the color of the pixel for which the ray hits the i-th sphere.
            # https://stackoverflow.com/questions/15619830/raytracing-how-to-combine-diffuse-and-specular-color
            if sphere_i_hit.any():
                hit_points_sphere_i = hit_points[sphere_i_hit]
                hit_normal_sphere_i = hit_normals[sphere_i_hit]
                hit_points_to_light = nn.functional.normalize(light_pos - hit_points_sphere_i, dim=-1)
                hit_points_to_camera = nn.functional.normalize(self.camera - hit_points_sphere_i, dim=-1)

                # Ambient color
                ambient = sphere.ambient

                # Diffuse color
                diffuse = sphere.diffuse * torch.clamp(torch.sum(hit_normal_sphere_i * hit_points_to_light, dim=-1, keepdim=True), min=0.0)

                # Specular color
                reflection = 2 * hit_normal_sphere_i * torch.sum(hit_normal_sphere_i * hit_points_to_light, dim=-1, keepdim=True) - hit_points_to_light
                specular = sphere.specular * torch.clamp(torch.sum(reflection * hit_points_to_camera, dim=-1, keepdim=True), min=0.0)**sphere.shininess

                # Combine color
                color = (ambient + diffuse + specular) * sphere.color * light_color
                color = torch.clamp(color, min=0.0, max=1.0)

                self.image[sphere_i_hit] = color

        return self.image.reshape(self.image_height, self.image_width, 3)


    # Peform ray-sphere hit calculation for all spheres and all rays at once.
    def ray_hit_sphere(self, 
                       ray_origins      : torch.Tensor,
                       ray_directions   : torch.Tensor,
                       sphere_centers   : torch.Tensor,
                       sphere_radiuses  : torch.Tensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate ray-sphere intersection.
        # Please check Section 5 in https://raytracing.github.io/books/RayTracingInOneWeekend.html.
        # However, I think the above link contains an error in calculating the value of b.
        # The method calculated using the link below seems more accurate.
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html
        oc = ray_origins[:, None, :] - sphere_centers[None, :, :]
        a = torch.sum(ray_directions**2, dim=-1, keepdim=True)          # torch.sum(..., dim=-1) for dot product calculation
        b = 2.0 * torch.sum(ray_directions[:, None, :] * oc, dim=-1)   # torch.sum(..., dim=-1) for dot product calculation
        c = torch.sum(oc**2, dim=-1) - sphere_radiuses**2               # torch.sum(..., dim=-1) for dot product calculation
        discriminant = b**2 - 4*a*c

        # Calculate the distance between the ray origin and the hit point.
        # discriminant >= 0 means there is at least one hit point.
        t = torch.where(discriminant >= 0, (-b - torch.sqrt(discriminant)) / (2.0 * a), float('inf'))

        min_t, min_idx = torch.min(t, dim=-1)
        valid_hit = min_t < float('inf')

        return min_t, min_idx, valid_hit


def main():
    ray_tracer = RayTracingModule(image_width=512, image_height=512)

    scene = [Sphere(center=[-2.5, 20.0, 2.5], radius=1.0, color=[1.0, 0.0, 0.0]),
            Sphere(center=[0.0, 20.0, 2.5], radius=1.0, color=[0.5, 1.0, 0.5]),
            Sphere(center=[2.5, 20.0, 2.5], radius=1.0, color=[0.5, 1.0, 0.0]),
            Sphere(center=[-2.5, 20.0, 0.0], radius=1.0, color=[1.0, 0.5, 0.0]),
            Sphere(center=[0.0, 20.0, 0.0], radius=1.0, color=[0.0, 0.5, 1.0]),
            Sphere(center=[2.5, 20.0, 0.0], radius=1.0, color=[0.5, 0.5, 1.0]),
            Sphere(center=[-2.5, 20.0, -2.5], radius=1.0, color=[1.0, 1.0, 1.0]),
            Sphere(center=[0.0, 20.0, -2.5], radius=1.0, color=[1.0, 1.0, 0.0]),
            Sphere(center=[2.5, 20.0, -2.5], radius=1.0, color=[1.0, 0.5, 0.5])]
    light_pos = torch.tensor([0.0, 16.0, 0.0], dtype=torch.float32).to("cuda")
    light_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to("cuda")

    import os
    if not os.path.exists("./rendered_images"):
        os.makedirs("./rendered_images")

    for i in range(240):
        light_pos[0] = 4.0 * math.sin(i * 0.05)
        light_pos[2] = 4.0 * math.cos(i * 0.05)
        image = ray_tracer(scene, light_pos, light_color)

        plt.imsave(f"./rendered_images/refactoring_rendered_image_{i:03}.jpg", image.cpu().numpy())

    print("rendering done!")

if __name__ == "__main__":
    main()