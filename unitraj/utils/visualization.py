import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def check_loaded_data(plt, data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    if len(agents.shape) == 4:
        agents = agents[index]
        map = map[index]
        ego_index = data['track_index_to_predict'][index]
        ego_agent = agents[ego_index]
    else:
        ego_index = data['track_index_to_predict']
        ego_agent = agents[ego_index]

    def draw_line_with_mask(point1, point2, color, line_width=4):
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Plot the map with mask check
    for lane in map:
        # map_one_hot = lane[0, -20:]
        # if np.argmax(map_one_hot) in [1, 2, 3]:
        #     continue
        for i in range(len(lane) - 1):
            draw_line_with_mask(lane[i, :2], lane[i, 6:8], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)

    # Set labels, limits, and other properties
    #vis_range = 100
    # plt.xlim(-vis_range + 30, vis_range + 30)
    # plt.ylim(-vis_range, vis_range)
   # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.axis('equal')
    #plt.tight_layout()

    return plt


def visualize_batch_data(ax, data):
    def decode_obj_trajs(obj_trajs):
        obj_trajs_xy = obj_trajs[..., :2]
        obj_lw = obj_trajs[...,-1, 3:5]
        obj_type_onehot = obj_trajs[...,-1, 6:9]
        obj_type = np.argmax(obj_type_onehot, axis=-1)
        obj_heading_encoding = obj_trajs[...,-1, 33:35]
        return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding
    def decode_map(map):
        map_xy = map[..., :2]
        map_type = map[...,0, 9:29]
        map_type = np.argmax(map_type, axis=-1)
        return map_xy, map_type

    def plot_objects(obj_xy, obj_lw, obj_heading, obj_mask):
        # 在已有的ax对象上进行绘制
        for i in range(len(obj_lw)):
            if obj_mask[i]:
                # 获取对象的长和宽
                length, width = obj_lw[i]

                # 通过sin和cos计算旋转角度
                sin_angle, cos_angle = obj_heading[i]
                angle = np.arctan2(sin_angle, cos_angle)  # 转换为角度（弧度）

                # 获取对象的中心位置
                x, y = obj_xy[i]

                # 创建旋转矩形对象
                rect = plt.Rectangle((-length / 2, -width / 2), length, width, angle=0,
                                     facecolor='none', edgecolor='grey', linewidth=1)

                # 使用转换矩阵将矩形旋转并平移到对象的中心位置
                t = ax.transData
                # 先将矩形平移到中心位置，然后旋转
                rot = plt.matplotlib.transforms.Affine2D().rotate_around(0, 0, angle).translate(x, y) + t
                rect.set_transform(rot)

                # 添加矩形到现有ax中
                ax.add_patch(rect)
    def draw_trajectory(trajectory, line_width, ego=False):
        def interpolate_color(start_color, end_color, t, total_t):
            """根据 t 和 total_t 插值计算颜色."""
            return [(1 - t / total_t) * start + (t / total_t) * end for start, end in zip(start_color, end_color)]

        def draw_line_with_mask(point1, point2, color, line_width=4):
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color,alpha=0.5)
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                # 天蓝色渐变：从深蓝到浅蓝
                start_color = (0, 0, 0.5)  # 深蓝色
                end_color = (0.53, 0.81, 0.98)  # 浅蓝色
            else:
                # 草绿色渐变：从深绿到浅绿
                start_color = (0, 0.5, 0)  # 深绿色
                end_color = (0.56, 0.93, 0.56)  # 浅绿色

            # 计算当前时间步的颜色
            color = interpolate_color(start_color, end_color, t, total_t)

            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    obj_trajs = data['obj_trajs']
    map = data['map_polylines']

    obj_trajs_xy, obj_lw, obj_type, obj_heading = decode_obj_trajs(obj_trajs)
    obj_trajs_future_state = data['obj_trajs_future_state'][...,:2]
    all_traj = np.concatenate([obj_trajs_xy, obj_trajs_future_state], axis=-2)

    for i in range(obj_trajs.shape[0]):
        if i == data['track_index_to_predict']:
            ego = True
        else:
            ego = False
        draw_trajectory(all_traj[i], line_width=3,ego=ego)

    map_xy, map_type = decode_map(map)
    obj_mask = data['obj_trajs_mask']
    plot_objects(obj_trajs_xy[:,-1],obj_lw, obj_heading, obj_mask[:,-1])

    for indx, type in enumerate(map_type):
        lane = map_xy[indx]
        if type == 0:
            continue
        if type in [1, 2, 3]:
            # 使用灰色虚线表示中心线
            color = 'grey'
            linestyle = 'dotted'
            linewidth = 1
        else:
            color = 'grey'
            linestyle = '-'
            linewidth = 0.2

        # 绘制线条
        for i in range(len(lane) - 1):
            if lane[i, 0] and lane[i + 1, 0]:
                ax.plot([lane[i, 0], lane[i + 1, 0]], [lane[i, 1], lane[i + 1, 1]],
                        linewidth=linewidth, color=color, linestyle=linestyle)

    # 设置坐标轴比例和范围
    vis_range = 35
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(True)
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    #plt.show()
    return ax

def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image


def visualize_prediction(batch, prediction, draw_index=0):
    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    batch = batch['input_dict']
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy()
    future_traj = batch['obj_trajs_future_state'][draw_index].cpu().numpy()
    past_traj_mask = batch['obj_trajs_mask'][draw_index].cpu().numpy()
    future_traj_mask = batch['obj_trajs_future_mask'][draw_index].cpu().numpy()
    pred_future_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy()
    pred_future_traj = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()

    map_xy = map_lanes[..., :2]

    map_type = map_lanes[..., 0, -20:]

    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1.5)

    # draw past trajectory
    for idx, traj in enumerate(past_traj):
        draw_trajectory(traj, line_width=2)

    # draw future trajectory
    for idx, traj in enumerate(future_traj):
        draw_trajectory(traj, line_width=2)

    # predicted future trajectory is (n,future_len,2) with n possible future trajectories, visualize all of them
    for idx, traj in enumerate(pred_future_traj):
        # calculate color based on probability
        color = cm.hot(pred_future_prob[idx])
        for i in range(len(traj) - 1):
            draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=2)

    return plt
