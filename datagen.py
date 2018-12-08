class DataGenerator:
    
    def create_train(dataset_ids, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_ids = shuffle(dataset_ids)
            for start in range(0, len(dataset_ids), batch_size):
                end = min(start + batch_size, len(dataset_ids))
                batch_images = []
                X_train_batch_ids = dataset_ids[start:end]
                batch_labels = np.zeros((len(X_train_batch_ids), N_CLASS))
                for i in range(len(X_train_batch_ids)):
                    image = DataGenerator.load_image(
                        X_train_batch_ids[i]['path'], shape)   
                    if augument:
                        image = DataGenerator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch_ids[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels
    
    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
        np.array(image_red_ch), 
        np.array(image_green_ch), 
        np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug