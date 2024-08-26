void encode_rows_cpu(THFloatTensor* input, THLongTensor* output);
void encode_cols_cpu(THFloatTensor* input, THLongTensor* output);
void encode_rows_cpu_4(THIntTensor* input, THLongTensor* output1, THLongTensor* output2, THLongTensor* output3, THLongTensor* output4);
void binary_gemm_cpu(THLongTensor* a, THLongTensor* b, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha, int alphas);
void binary_gemm_cpu_4(THLongTensor* a1, THLongTensor* a2, THLongTensor* a3, THLongTensor* a4, THLongTensor* b, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha);
void folder(THFloatTensor* input_unfolder, THFloatTensor* input, int* mask, int C, int N, int T, int H, int H_p, int kernel_size, int padding, int stride);
void unfolder(THFloatTensor* input, THFloatTensor* input_unfolder, int kernel_size[2], int N, int T, int H, int C, int stride);
void binary_gemm_cpu_4_activation(THLongTensor* a1, THLongTensor* a2, THLongTensor* a3, THLongTensor* a4, THLongTensor* b1, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha, THFloatTensor* alphas);
void mm(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int nn, int k);
void dot_add(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int n);
void full_mm(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int n, int k);
void THNN_Bin_SpatialConvolutionMM_updateOutput(
          THFloatTensor *input,
          THFloatTensor *output,
          THIntTensor *weight,
          THFloatTensor *bias,
          THFloatTensor *columns,
          THFloatTensor *alphas,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW);