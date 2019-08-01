def remove_item(img, left, top):
    fridge = mpimg.imread('fridge.jpg')
    for r in range(top, top+80):
        for c in range(left, left+80):
            img[r][c] = fridge[r][c]
    return img