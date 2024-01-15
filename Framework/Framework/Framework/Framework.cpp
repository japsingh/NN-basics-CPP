#include <iostream>
#include "MatrixOps.h"
#include "NN.h"

void MatrixTest()
{
    //Matrix<int> m({ 3,2 });
    //m.set({ 0,0 }, 101);
    //m.set({ 0,1 }, 102);
    //m.set({ 1,0 }, 103);
    //m.set({ 1,1 }, 104);
    //m.set({ 2,0 }, 105);
    //m.set({ 2,1 }, 106);
    //m.print();
    ////std::cout << m.get({ 0,0 });

    //Matrix<int> m2({ 3 });
    //m2.set({ 0 }, 101);
    //m2.set({ 1 }, 102);
    //m2.set({ 2 }, 103);
    //m2.print();

    //Matrix<int> m3({ 4,3,2 });
    //m3.set({ 0,0,0 }, 101);
    //m3.set({ 0,0,1 }, 102);
    //m3.set({ 0,1,0 }, 103);
    //m3.set({ 0,1,1 }, 104);
    //m3.set({ 0,2,0 }, 105);
    //m3.set({ 0,2,1 }, 106);

    //m3.set({ 1,0,0 }, 1101);
    //m3.set({ 1,0,1 }, 1102);
    //m3.set({ 1,1,0 }, 1103);
    //m3.set({ 1,1,1 }, 1104);
    //m3.set({ 1,2,0 }, 1105);
    //m3.set({ 1,2,1 }, 1106);

    //m3.set({ 2,0,0 }, 2101);
    //m3.set({ 2,0,1 }, 2102);
    //m3.set({ 2,1,0 }, 2103);
    //m3.set({ 2,1,1 }, 2104);
    //m3.set({ 2,2,0 }, 2105);
    //m3.set({ 2,2,1 }, 2106);

    //m3.set({ 3,0,0 }, 3101);
    //m3.set({ 3,0,1 }, 3102);
    //m3.set({ 3,1,0 }, 3103);
    //m3.set({ 3,1,1 }, 3104);
    //m3.set({ 3,2,0 }, 3105);
    //m3.set({ 3,2,1 }, 3106);

    //m3.print();

    //Matrix<float> m4 = RandomFloatMatrix({ 2,2 }, 10.f, 20.f);
    //m4.print();

    Matrix2d<int> m5 = RandomIntMatrix2d({ 5,4 }, 0, 10);
    m5.print();

    Matrix2d<int> m6 = RandomIntMatrix2d({ 4,5 }, 0, 10);
    m6.print();

    //m6.sum(m5);
    //m6.print();

    Matrix2d<int> m7(Matrix2d<int>::get_mult_dst_shape(m5, m6));
    MatMul2d(m7, m5, m6);
    m7.print();
}

static float ep = 1e-3f;
static float lr = 1e-1f;

static const std::vector<std::vector<float>> train_xor = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};
static const std::vector<std::vector<float>> train_or = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};

void XorTest()
{
    //Xor model;
    NN model;
    // Add input layer
    model.append_layer(2);
    // Add hidden layer
    model.append_layer(2);
    // Add output layer
    model.append_layer(1);

    Matrix2d<float> train(train_xor);
    Matrix2d<float> X;
    Matrix2d<float> Y;
    SplitXY(train, X, Y);

    size_t epochs = 1000*15;
    for (size_t i = 0; i < epochs; ++i) {
        model.train(ep, lr, X, Y);
        if ((i % 100) == 0) {
            std::cout << "Epochs: " << i << ", Cost: " << model.cost(X, Y) << std::endl;
        }
    }
    std::cout << "Final Cost: " << model.cost(X, Y) << std::endl;
    for (size_t r = 0; r < X.rows(); ++r) {
        Matrix2d<float> row = X.get_row(r);
        float x1 = row.get({ 0,0 });
        float x2 = row.get({ 0,1 });
        model.input(row);
        model.forward();
        auto output = model.output();
        float yhat = output.get({ 0,0 });
        std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
    }

}

int main()
{
    //MatrixTest();
    XorTest();
}