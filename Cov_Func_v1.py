import numpy as np

def Cov_Func(pop, rs, theta0, Obstacle_Area, Covered_Area):
    Covered_Area[Covered_Area != 0] = 0

    # khởi tạo lưới toạ độ
    size_x, size_y = Covered_Area.shape
    X, Y = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing='ij')

    inside_sector = np.zeros_like(Covered_Area, dtype=bool)

    for j in range(pop.shape[0]):
        x0 = pop[j, 0]
        y0 = pop[j, 1]
        alpha = pop[j, 2]
        r = rs[j]
        theta = theta0[j]

        # tính khoảng cách và góc
        D = np.sqrt((X - x0)**2 + (Y - y0)**2)
        Theta = np.arctan2(Y - y0, X - x0)
        Theta[Theta < 0] += 2 * np.pi

        # kiểm tra điều kiện trong hình quạt
        in_circle = D <= r
        if alpha - theta / 2 < 0:
            in_angle = (Theta >= (alpha - theta / 2 + 2 * np.pi)) | (Theta <= (alpha + theta / 2))
        elif alpha + theta / 2 > 2 * np.pi:
            in_angle = (Theta >= (alpha - theta / 2)) | (Theta <= (alpha + theta / 2 - 2 * np.pi))
        else:
            in_angle = (Theta >= (alpha - theta / 2)) & (Theta <= (alpha + theta / 2))

        inside_sector |= (in_circle & in_angle)

    # cập nhật vùng phủ
    Covered_Area[inside_sector] = Obstacle_Area[inside_sector]

    # xử lý trùng lên vật cản
    mask_obstacle = (Obstacle_Area == 0) & (Covered_Area == 1)
    Covered_Area[mask_obstacle] = -2

    # tính độ phủ
    count1 = np.count_nonzero(Covered_Area == 1)
    count2 = np.count_nonzero(Covered_Area == -2)
    count3 = np.count_nonzero(Obstacle_Area == 1)
    coverage = (count1 - count2) / count3 if count3 > 0 else 0

    # khôi phục vùng bị vật cản phủ thành -1
    Covered_Area[Covered_Area == -2] = -1

    return coverage, Covered_Area
