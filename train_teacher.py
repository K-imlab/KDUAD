from DataLoader import DataLoader, CustomDataset, load_raw
from model import AE
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import utils


def rmse_loss(recon, origin):
    n_feature = origin.shape[-1]

    # calculate rmse
    error = tf.math.subtract(recon, origin)
    error = tf.math.pow(error, 2)
    error = tf.math.reduce_sum(error, axis=1)
    error = tf.math.divide(error, n_feature)
    error = tf.math.sqrt(error)
    # calculate mean of rmse value by batch for train
    error_mean = tf.reduce_mean(error)

    # calculate median and maximum of rmse value by batch for test
    error_array = np.array(error)
    error_median = np.median(error_array)
    error_maximum = np.max(error_array)
    error_minimum = np.min(error_array)
    return error_mean, error_median, error_maximum, error_minimum


def train_step_usad(train_x, epoch):
    with tf.GradientTape(persistent=True) as tape:
        w1, w2, w3 = teacher_AE(train_x)
        loss1 = usad_loss(step=1, recon=w1, rerecon=w3, origin=train_x, n=epoch + 1)

    gradients_enc = tape.gradient(loss1, teacher_AE.encoder.trainable_variables)
    gradients_dec = tape.gradient(loss1, teacher_AE.decoder.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients_enc, teacher_AE.encoder.trainable_variables))
    OPTIMIZER.apply_gradients(zip(gradients_dec, teacher_AE.decoder.trainable_variables))

    with tf.GradientTape(persistent=True) as tape:
        w1, w2, w3 = teacher_AE(train_x)
        loss2 = usad_loss(step=2, recon=w2, rerecon=w3, origin=train_x, n=epoch + 1)

    gradients_enc = tape.gradient(loss2, teacher_AE.encoder.trainable_variables)
    gradients_dec = tape.gradient(loss2, teacher_AE.decoder2.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients_enc, teacher_AE.encoder.trainable_variables))
    OPTIMIZER.apply_gradients(zip(gradients_dec, teacher_AE.decoder2.trainable_variables))

    return loss1, loss2


def val_step_usad(val_x, epoch) :
    w1, w2, w3 = teacher_AE(val_x)

    loss1 = usad_loss(1, w1, w3, val_x, epoch+1)
    loss2 = usad_loss(2, w2, w3, val_x, epoch+1)

    return loss1, loss2


def usad_loss(step, recon, rerecon, origin, n=1, a=1) :
    loss1, loss1_median, loss1_max, loss1_min = rmse_loss(recon, origin)
    loss2, loss2_median, loss2_max, loss2_min = rmse_loss(rerecon, origin)

    # Step teacher : Train
    # Step 2 : Validation
    # Step 3 : Test
    if step == 1:
        loss = tf.abs(((1/n) * loss1) + ((1-(1/n))*loss2))

        return loss

    elif step == 2:
        loss = tf.abs(((1/n) * loss1) - ((1-(1/n))*loss2))

        return loss

    elif step == 3:
        mean = (a * loss1) + ((1-a) * loss2)
        median = (a * loss1_median) + ((1-a) * loss2_median)
        max = (a * loss1_max) + ((1-a) * loss2_max)
        min = (a * loss1_min) + ((1-a) * loss2_min)

        return mean.numpy(), median, max, min


if __name__ == "__main__":

    save_path = "result/teacher"
    teacher_n_input = 52
    student_n_input = 18
    batch_size = 256
    epochs = 200
    lr = 0.001

    teacher_n_features = [teacher_n_input, 256, 128, 64, 32, 18]
    student_n_features = [student_n_input, 256, 128, 64, 32, 18]

    teacher_AE = AE(teacher_n_features)
    student_AE = AE(student_n_features)
    OPTIMIZER = Adam(learning_rate=lr)

    train_X, val_X, train_y, val_y, test, additional_test = load_raw()

    train_set = CustomDataset(train_X, train_y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = CustomDataset(val_X, val_y)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    test_X = CustomDataset(test, None)
    test_loader = DataLoader(test_X, batch_size=batch_size, shuffle=True)

    label_test = CustomDataset(additional_test, pd.DataFrame(np.ones_like(additional_test)))
    label_test_loader = DataLoader(label_test, batch_size=batch_size, shuffle=True)

    results = {'train_loss': [], 'val_loss': [], 'train_loss1': [],
               'train_loss2': [], 'val_loss1': [], 'val_loss2': []}

    for epoch in range(epochs):
        train_loss = []
        train_loss2 = []

        for i, data in enumerate(train_loader):
            teacher_train_data, label = data
            loss1, loss2 = train_step_usad(teacher_train_data, epoch)
            print("\rTraining : {} / {} ".format(i + 1, len(train_loader)), end="")
            train_loss.append(loss1)
            train_loss2.append(loss2)

        train_loader.on_epoch_end()
        print("\tTraining is completed...")

        val_loss = []
        val_loss2 = []
        for j, data in enumerate(val_loader):
            teacher_val_data, label = data
            loss1, loss2 = val_step_usad(teacher_val_data, epoch)
            print("\rValidation : {} / {}".format(j + 1, len(val_loader)), end="")
            val_loss.append(loss1)
            val_loss2.append(loss2)

        val_loader.on_epoch_end()
        print("\tValidation is completed...")

        train_loss_avg = sum(train_loss) / len(train_loss)
        val_loss_avg = sum(val_loss) / len(val_loss)
        train_loss2_avg = sum(train_loss2) / len(train_loss2)
        val_loss2_avg = sum(val_loss2) / len(val_loss2)
        results['train_loss1'].append(train_loss_avg.numpy())
        results['train_loss2'].append(train_loss2_avg.numpy())
        results['val_loss1'].append(val_loss_avg.numpy())
        results['val_loss2'].append(val_loss2_avg.numpy())

        teacher_AE.save_weights(os.path.join(save_path, f"{epoch + 1: 05d} epoch_weights"))

        if epoch > 0:
            if (val_loss_avg + val_loss2_avg) / 2 < min(results['val_loss']):
                teacher_AE.save_weights(os.path.join(save_path, "Best_weights"))

            if val_loss_avg.numpy() < min(results['val_loss']):
                teacher_AE.save_weights(os.path.join(save_path, "Best_weights"))

        results['train_loss'].append(((train_loss_avg + train_loss2_avg) / 2).numpy())
        results['val_loss'].append(((val_loss_avg + val_loss2_avg) / 2).numpy())

        print(
            "{:>3} / {:>3} || train_loss: {:8.4f}, val_loss: {:8.4f}".format(
                epoch + 1, epochs,
                results['train_loss'][-1],
                results['val_loss'][-1], ))
        print("_"*30)
        # early stop
        if epoch > 40:
            if results['val_loss'][-11] < min(results['val_loss'][-10:]):
                print(results['val_loss'][-11])
                print(min(results['val_loss'][-11:]))
                break

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_path, 'train_results.csv'), index=False)

    alpha = 0.9
    for x, label in label_test_loader:
        w1, w2, w3 = teacher_AE.predict(x)
        mean, median, maximum, minimum = usad_loss(step=3, recon=w1, rerecon=w3, origin=x, a=alpha)

        print(mean.shape, label.shape)

        opt_threshold = utils.draw_roc(label, mean)
        accuracy, f1, recall, precision = utils.get_metric(label, mean, opt_threshold)
        print(accuracy, f1, recall, precision)