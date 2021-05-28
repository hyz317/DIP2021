import argparse
import numpy as np
import random
import torch
from sklearn.svm import SVC

from dataset.dataset import DataSet
from model_regression.model import RegressionModel


def train(model, count, optimizer):
    model.train()
    for i in range(count):
        optimizer.zero_grad()
        class_idx = random.randint(1, 900)
        posi_num = random.sample(list(range(1,10))+list(range(10,105,5)), 1)[0]
        # small set
        subset_x, subset_y = dataset.getBaseSubset(class_idx, posi_num, 5)
        small_model = SVC(kernel = 'linear', random_state = 0, tol=random.sample([0.01,0.1,1,10,100], 1)[0])
        small_model.fit(subset_x, subset_y)
        w = torch.tensor(small_model.coef_.copy(), dtype=torch.float32)
        b = torch.tensor(small_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
        w0 = torch.cat([w, b], axis=-1).cuda()

        # big set
        bigset_x, bigset_y = dataset.getBaseSubset(class_idx, 100, 100)
        big_model = SVC(kernel = 'linear', random_state = 0)
        big_model.fit(bigset_x, bigset_y)
        w2 = torch.tensor(big_model.coef_.copy(), dtype=torch.float32)
        b2 = torch.tensor(big_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
        wstar = torch.cat([w2, b2], axis=-1).cuda()

        # transform
        Tw0 = model(w0)

        # loss
        valid_subset_x, valid_subset_y = dataset.getBaseSubset(class_idx, posi_num, 100)
        loss1 = 0.5 * torch.sum((Tw0 - wstar) * (Tw0 - wstar))
        
        predict = torch.matmul(valid_subset_x.cuda(), Tw0[0, :4096]) + Tw0[0, 4096]
        predict *= valid_subset_y.cuda()
        predict[predict > 1] = 1
        loss2 = torch.sum(1 - predict, 0)
        
        lam = 1
        loss = loss1 + lam * loss2
        predict[predict < 0] = 0
        predict[predict > 0] = 1
        print("count={}, loss1={:.4f}, loss2={:.4f}, acc={:.4f}".format(i, loss1, loss2, torch.sum(predict)/predict.shape[0]))

        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            optimizer.zero_grad()
            class_idx = random.randint(901, 1000)
            posi_num = random.sample(list(range(1,10))+list(range(10,105,5)), 1)[0]
            # small set
            subset_x, subset_y = dataset.getBaseSubset(class_idx, posi_num, 5)
            small_model = SVC(kernel = 'linear', random_state = 0, tol=random.sample([0.01,0.1,1,10,100], 1)[0])
            small_model.fit(subset_x, subset_y)
            w = torch.tensor(small_model.coef_.copy(), dtype=torch.float32)
            b = torch.tensor(small_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
            w0 = torch.cat([w, b], axis=-1).cuda()

            # big set
            bigset_x, bigset_y = dataset.getBaseSubset(class_idx, 100, 100)
            big_model = SVC(kernel = 'linear', random_state = 0)
            big_model.fit(bigset_x, bigset_y)
            w2 = torch.tensor(big_model.coef_.copy(), dtype=torch.float32)
            b2 = torch.tensor(big_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
            wstar = torch.cat([w2, b2], axis=-1).cuda()

            # transform
            Tw0 = model(w0)

            # loss
            valid_subset_x, valid_subset_y = dataset.getBaseSubset(class_idx, posi_num, 100)
            loss1 = 0.5 * torch.sum((Tw0 - wstar) * (Tw0 - wstar))
            
            predict = torch.matmul(valid_subset_x.cuda(), Tw0[0, :4096]) + Tw0[0, 4096]
            predict *= valid_subset_y.cuda()
            predict[predict > 1] = 1
            loss2 = torch.sum(1 - predict, 0)
            
            lam = 1
            loss = loss1 + lam * loss2
            predict[predict < 0] = 0
            predict[predict > 0] = 1
            print("VALID count={}, loss1={:.4f}, loss2={:.4f}, acc={:.4f}".format(i, loss1, loss2, torch.sum(predict)/predict.shape[0]))
            p = small_model.predict(valid_subset_x) * valid_subset_y.numpy()
            p[p < 0] = 0
            print("VALID Before: acc={:.4f}".format(np.sum(p) / p.shape[0]))


def refine(model, count, novel_class_id):
    model.train()
    # init
    subset_x, subset_y = dataset.getNovelSubset(novel_class_id, random.randint(3,10), 5)
    init_model = SVC(kernel = 'linear', random_state = 0, tol=random.sample([0.01,0.1,1,10,100], 1)[0])
    init_model.fit(subset_x, subset_y)
    w = torch.tensor(init_model.coef_.copy(), dtype=torch.float32)
    b = torch.tensor(init_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
    W = torch.cat([w, b], axis=-1).cuda()
    model.W = torch.nn.Parameter(W)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(count):
        optimizer.zero_grad()
        # small set
        subset_x, subset_y = dataset.getNovelSubset(novel_class_id, random.randint(3,10), 5)
        small_model = SVC(kernel = 'linear', random_state = 0, tol=random.sample([0.01,0.1,1,10,100], 1)[0])
        small_model.fit(subset_x, subset_y)
        w = torch.tensor(small_model.coef_.copy(), dtype=torch.float32)
        b = torch.tensor(small_model.intercept_.copy(), dtype=torch.float32).reshape([1, 1])
        w0 = torch.cat([w, b], axis=-1).cuda()

        # transform
        Tw0 = model(w0)

        # loss
        loss1 = 0.5 * torch.sum((Tw0 - model.W) * (Tw0 - model.W))
        
        predict = torch.matmul(subset_x.cuda(), model.W[0, :4096]) + model.W[0, 4096]
        predict *= subset_y.cuda()
        predict[predict > 1] = 1
        loss2 = torch.sum(1 - predict, 0)
        
        lam = 1
        loss = loss1 + lam * loss2
        predict[predict < 0] = 0
        predict[predict > 0] = 1
        print("refine count={}, loss1={:.4f}, loss2={:.4f}, acc={:.4f}".format(i, loss1, loss2, torch.sum(predict)/predict.shape[0]))
        # print(model.W)

        loss.backward()
        optimizer.step()


    # test
    predict = torch.matmul(dataset.test_set["features"].cuda(), model.W[0, :4096]) + model.W[0, 4096]
    for i in range(len(predict)):
        if predict[i] > 0:
            print("result! {}, {:.4f}".format(dataset.test_set["names"][i], predict[i]))

    predict = torch.matmul(dataset.test_set["features"].cuda(), w0[0, :4096]) + w0[0, 4096]
    for i in range(len(predict)):
        if predict[i] > 0:
            print("origin result! {}, {:.4f}".format(dataset.test_set["names"][i], predict[i]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="data/training_224x224")
    parser.add_argument('--base_dir', type=str, default="data/base")
    parser.add_argument('--test_set_dir', type=str, default="data/testing_224x224")
    args = parser.parse_args()

    dataset = DataSet(
        training_img_dir=args.train_set_dir,
        test_img_dir=args.test_set_dir,
        base_info_dir=args.base_dir
    )

    model = RegressionModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, 100, optimizer)
    refine(model, 70, 1)
    
