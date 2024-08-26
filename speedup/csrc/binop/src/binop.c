#include <TH/TH.h>
#include <stdio.h>
#include <stdint.h>
#include "matmul.h"
#include <time.h>

inline uint64_t encode_val(float* array, int n) {
    uint64_t sign, r = 0;
    for(int i=0; i<ENCODE_BIT && i<n; i++){
        sign = array[i]>0;
        r |= (sign<<i);
    }
    return r;
}
inline __m256i encode_val_256(uint64_t* array, int n) {
    __m256i r;
    uint64_t a[4] = {0,0,0,0};
    for(int i=0; i<4 && i<n; i++){
        a[4 - i -1] = array[i];    
    }

    r = _mm256_set_epi64x(a[0], a[1], a[2], a[3]);

    return r;
}
inline void encode_rows_cpu_kernel_256(uint64_t *columns, __m256i *columns_binary, int m, int n) {//这里的n代表的是第一次压缩后的n
    int i, l = 1+(n-1)/4;        

    //#pragma omp parallel for
    for (i = 0; i < m*l; i++) {
        long long p = n*(i/l)+4*(i%l);
        
        columns_binary[i] = encode_val_256(&columns[p], n-4*(i%l));

        //printf('%d',i);
    }

}
inline void encode_val_4(int* array, int n, uint64_t * r) {
    uint64_t sign, flag;
    for(int i=0; i<ENCODE_BIT && i<n; i++){
        flag = array[i];

        for(int j = 0; j < 4; j++){
            sign = flag % 2 !=0;
            flag = flag / 2;
            r[j] |= (sign<<i);            
        }

    }
}
void encode_rows_cpu_kernel(float *columns, uint64_t *columns_binary, int m, int n) {
    int i;
    size_t l = 1+(n-1)/ENCODE_BIT;
    //#pragma omp parallel for
    for (i = 0; i < m*l; i++) {
        size_t p = n*(i/l)+ENCODE_BIT*(i%l);
        columns_binary[i] = encode_val(&columns[p], n-ENCODE_BIT*(i%l));
    }
}
void encode_rows_cpu_kernel_4(int *columns, uint64_t *columns_binary1, uint64_t *columns_binary2, uint64_t *columns_binary3, uint64_t *columns_binary4, int m, int n) {
    int i;
    size_t l = 1+(n-1)/ENCODE_BIT;
    uint64_t a[4], r[4]={0,0,0,0};
    //#pragma omp parallel for
    for (i = 0; i < m*l; i++) {
        size_t p = n*(i/l)+ENCODE_BIT*(i%l);

        //columns_binary[i] = encode_val(&columns[p], n-ENCODE_BIT*(i%l));
        encode_val_4(&columns[p], n-ENCODE_BIT*(i%l), r);
        columns_binary1[i] = r[0]; 
        r[0] = 0;
        columns_binary2[i] = r[1];
        r[1] = 0;
        columns_binary3[i] = r[2];
        r[2] = 0;
        columns_binary4[i] = r[3];
        r[0] = 0;
    }
}
void encode_cols_cpu_kernel(float *columns, uint64_t *columns_binary, int m, int n) {
    int col_bin_m = 1 + (m-1) / ENCODE_BIT;
    int i, j, k;
    //#pragma omp parallel for
    for (i = 0; i < col_bin_m; i++) {
        int i64 = i * ENCODE_BIT;
        for (j = 0; j < n && i64<m ; j++) {

            uint64_t sign, rvalue = 0;

            for (k = 0; j + n * (i64 + k) < m*n && k < ENCODE_BIT; k++) {
                sign = columns[j + n * (i64 + k)]>0;
                rvalue |= (sign << k);
            }

            columns_binary[j + n * i] = rvalue;
        }
    }
}

void encode_rows_cpu(THFloatTensor* input, THLongTensor* output) {
    int m = input->size[0];
    int n = input->size[1];
    int l = 1+(n-1)/ENCODE_BIT;

    THLongTensor_resize2d(output, m, l);
    float* a = THFloatTensor_data(input);
    uint64_t* b = (uint64_t*)THLongTensor_data(output);
    printf("%d %d\n",m, l);
    encode_rows_cpu_kernel(a, b, m, n);
}
void encode_rows_cpu_4(THIntTensor* input, THLongTensor* output1, THLongTensor* output2, THLongTensor* output3, THLongTensor* output4) {
    int m = input->size[0];
    int n = input->size[1];
    int l = 1+(n-1)/ENCODE_BIT;

    THLongTensor_resize2d(output1, m, l);
    THLongTensor_resize2d(output2, m, l);
    THLongTensor_resize2d(output3, m, l);
    THLongTensor_resize2d(output4, m, l);
    int* a = THIntTensor_data(input);
    uint64_t* b1 = (uint64_t*)THLongTensor_data(output1);
    uint64_t* b2 = (uint64_t*)THLongTensor_data(output2);
    uint64_t* b3 = (uint64_t*)THLongTensor_data(output3);
    uint64_t* b4 = (uint64_t*)THLongTensor_data(output4);

    encode_rows_cpu_kernel_4(a, b1, b2, b3, b4, m, n);
}
void encode_cols_cpu(THFloatTensor* input, THLongTensor* output) {
    int n = input->size[0];
    int k = input->size[1];
    int l = 1+(n-1)/ENCODE_BIT;

    THLongTensor_resize2d(output, l, k);
    float* a = THFloatTensor_data(input);
    uint64_t* b = (uint64_t*)THLongTensor_data(output);

    encode_cols_cpu_kernel(a, b, n, k);
}
void unfolder_kernel(float* input, float* input_unfolder, int kernel_size[2], int N, int T, int H, int C, int stride) {
    int i, j, h, w;
    for (i = 0; i < N; i++) {
        for (j = 0; j < T; j++) {
            h = j / H;
            w = j % H;
            
            int input_offset = i * C * H * H;
            int output_offset = i * T * kernel_size[0] * kernel_size[1];


            for (int c = 0; c < C; c++) {
                for (int kh = 0; kh < kernel_size[0]; kh++) {
                    for (int kw = 0; kw < kernel_size[1]; kw++) {
                        int input_row = h * stride + kh;
                        int input_col = w * stride + kw;
                        int input_index = input_offset + c * H * H + input_row * H + input_col;
                        int output_index = output_offset + c * kernel_size[0] * kernel_size[1] + kh * kernel_size[1] + kw;
                        
                        input_unfolder[output_index] = input[input_index];
                        
                    }
                }
            }
        }
    }
}
void folder_kernel(float* input_unfolder, float* input, int* mask, int C, int N, int T, int H, int H_p, int kernel_size, int padding, int stride) {
    int index = 0;
    for(int i =0; i < N; i++){
        if (mask[i]){
            int output_offset = i * C * H_p * H_p;
            for (int t = 0; t < T; t++){
                int input_offset = index * C * kernel_size * kernel_size;
                int h = (T / H) * stride; 
                int w = (T % H) * stride;
                for (int c = 0; c < C; c++){
                    int output_offset_ = output_offset + c * H_p * H_p;
                    int input_offset_ = input_offset + c * kernel_size * kernel_size;
                    for(int k1 = 0; k1 < kernel_size; k1++){
                        for(int k2 = 0; k2 < kernel_size; k2++){
                            int output_index = output_offset_ + (h + k1) * H_p + (w + k2);
                            int input_index = input_offset_ + k1 * kernel_size + k2;
                            input[output_index] += input_unfolder[input_index];
                        }
                    }
                }
                index++;
            }            
        }

    }

}
void mm_kernel(float *A, float *B, float *C, int m, int n, int k){
    for(int i=0; i<m; i++){
        for(int j = 0; j<k; j++){
            
            C[i * k + j] = A[i] * B[j]; 
            
        }
    }
}
void mm(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int nn, int k){
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }    
    float *A = (float*)THFloatTensor_data(a);
    float *B = (float*)THFloatTensor_data(b);
    float *C = THFloatTensor_data(c);   
    mm_kernel(A, B, C, m, nn, k); 
}
void dot_kernel(float *A, float *B, float *C, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            A[i * n + j] *= B[i];
            A[i * n + j] += C[i*n + j];
        }
    }
}
void dot_add(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int n){
    float *A = (float*)THFloatTensor_data(a);
    float *B = (float*)THFloatTensor_data(b);  
    float *C = (float*)THFloatTensor_data(c); 
    dot_kernel(A, B, C, m, n); 
}
void full_mm_kernel(float *A, float *B, float *C, int m, int n, int col){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < col; j++){
            int temp = 0;
            for(int k = 0; k < n; k++){
                temp += (A[i * n + k] * B[j * n + k]);
            }
            C[i * col + j] = temp;
        }
    }
}
void full_mm_block_kernel(float *A, float *B, float *C, int m, int n, int col){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < col; j++){
            int temp = 0;
            for(int k = 0; k < n; k++){
                temp += (A[i * n + k] * B[j * n + k]);
            }
            C[i * col + j] = temp;
        }
    }
}
void full_mm(THFloatTensor* a, THFloatTensor* b, THFloatTensor* c, int m, int n, int k){
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }    
    float *A = (float*)THFloatTensor_data(a);
    float *B = (float*)THFloatTensor_data(b);
    float *C = THFloatTensor_data(c);   
    full_mm_kernel(A, B, C, m, n, k);    
}
void unfolder(THFloatTensor* input, THFloatTensor* input_unfolder, int kernel_size[2], int N, int T, int H, int C, int stride){
    float *c = THFloatTensor_data(input);
    float *d = THFloatTensor_data(input_unfolder);
    unfolder_kernel(c, d, kernel_size, N, T, H, C, stride);

}
void folder(THFloatTensor* input_unfolder, THFloatTensor* input, int* mask, int C, int N, int T, int H, int H_p, int kernel_size, int padding, int stride){
    float *c = THFloatTensor_data(input_unfolder);
    float *d = THFloatTensor_data(input);
    folder_kernel(c, d, mask, C, N, T, H, H_p, kernel_size, padding, stride);
}
int min(int a, int b){
    if(a < b){
        return a;
    }
    else{
        return b;
    }
}
void pack(uint64_t* A, uint64_t* A_, int m, int n){
    for(int i = 0; i < m*n; i++){
        *A_++ = A[i];
    }
}
void binary_gemm_cpu(THLongTensor* a, THLongTensor* b, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha, int alphas){
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }
    uint64_t *A = (uint64_t*)THLongTensor_data(a);
    uint64_t *B = (uint64_t*)THLongTensor_data(b);
    
    size_t alignment = 8;
    float *C = THFloatTensor_data(c);
    //float *D = THFloatTensor_data(alphas);
    int n = 1 + (nn-1) / ENCODE_BIT, brow = transb? 1:k, bcol = transb? n:1;
    size_t size = 4 * n;

    dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, alphas);

}
void binary_gemm_cpu_4(THLongTensor* a1, THLongTensor* a2, THLongTensor* a3, THLongTensor* a4, THLongTensor* b, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha){
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }
    uint64_t *A1 = (uint64_t*)THLongTensor_data(a1);
    uint64_t *A2 = (uint64_t*)THLongTensor_data(a2);
    uint64_t *A3 = (uint64_t*)THLongTensor_data(a3);
    uint64_t *A4 = (uint64_t*)THLongTensor_data(a4);
    uint64_t *B = (uint64_t*)THLongTensor_data(b);
    float *C = THFloatTensor_data(c);
    float *C1 = (float *)malloc(m * k * sizeof(float));
    float *C2 = (float *)malloc(m * k * sizeof(float));
    float *C3 = (float *)malloc(m * k * sizeof(float));
    float *C4 = (float *)malloc(m * k * sizeof(float));
    //float C1[m*k], C2[m*k],C3[m*k],C4[m*k];
    //float *D = THFloatTensor_data(alphas);
    int n = 1 + (nn-1) / ENCODE_BIT, brow = transb? 1:k, bcol = transb? n:1;
    dgemm_nn(m, k, nn, A1, n, 1, B, brow, bcol, C1, k, 1, beta, 0, 1);
    //MM_4x4_block(A1, B, C1, m, n, k);
    dgemm_nn(m, k, nn, A2, n, 1, B, brow, bcol, C2, k, 1, beta, 1, 2);
    //MM_4x4_block(A2, B, C2, m, n, k);
    dgemm_nn(m, k, nn, A3, n, 1, B, brow, bcol, C3, k, 1, beta, 1, 4);
    //MM_4x4_block(A3, B, C3, m, n, k);
    dgemm_nn(m, k, nn, A4, n, 1, B, brow, bcol, C4, k, 1, beta, 1, 8);
    //MM_4x4_block(A4, B, C4, m, n, k);
    sum(C1, C2, C3, C4, C, m, k);
    //dgemm_nn_ours(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
    free(C1);
    free(C2);
    free(C3);
    free(C4);

}
void sum(float* C1, float* C2, float* C3, float* C4, float* C, int m, int k){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            C[i * k + j] = C1[i * k + j] + C2[i * k + j] + C3[i * k + j] + C4[i * k + j];
        }
    }
}
void normal_multiply_ijk_4x4(uint64_t *A, uint64_t *B, float *C, int ppm, int n, int ppcol, int col) {
    int x1 = -col, x2 = -n, x3 = -n;
    for (int i = 0; i < ppm; i++) {
        x1 += col;
        x2 += n;
        x3 = -n;
        for (int j = 0; j < ppcol; j++) {
            x3 += n;
            for (int k = 0; k < n; k++) {
                //C[i * col + j] += A[i * n + k] * B[j * n + k];
                C[x1 + j] -= (popcnt64(MASK(A[x2 + k]^B[x3 + k]))<<1);
            }
        }
    }
}
void MM_4x4_block(uint64_t *A, uint64_t *B, float *C, int m, int n, int col, uint64_t *A_, uint64_t *B_){
    int mc = 64, kc = 64;
    for(int i = 0; i < m; i += mc){
        int pm = min(m - i, mc);
        for(int j = 0; j < col; j += kc){
            int pk = min(col - j, kc);
            //printf("%d %d\n", i,j);
            MM_4x4_Inner(&A[i * n], &B[j * n], &C[i * col + j], pm, n, pk, m, n, col, A_, B_);
        }
    }
}
void MM_4x4_Inner(uint64_t *A, uint64_t *B, float *C, int pm, int pn, int pcol, int m, int n, int col, uint64_t *A_, uint64_t *B_){
    for(int j = 0; j < pcol; j+=4){
        for(int i = 0; i < pm; i+=4){
            if(j + 4 > pcol || i + 4 > pm){
                int ppcol = min(pcol - j, 4);
                int ppm = min(pm - i, 4);
                normal_multiply_ijk_4x4(&A[i * n], &B[j * n], &C[i * col + j], ppm, n, ppcol, col);
            }
            else{
                pack(&A[i * n], &A_[0], 4, n);
                pack(&B[j * n], &B_[0], 4, n);
                
                AddDot4x4(&A_[0], &B_[0], &C[i * col + j], m, n, col);
            }
            

        }
    }    
}
void AddDot4x4_SIMD_1(uint64_t *A, uint64_t *B, float *C, int m, int n, int col) {
  int p;
  register uint64_t a_0p_reg_re, a_1p_reg_re, a_2p_reg_re, a_3p_reg_re, b_p0_reg_re, b_p1_reg_re, b_p2_reg_re,
      b_p3_reg_re;
  register float c_00_reg_re,
      c_01_reg_re, c_02_reg_re, c_03_reg_re, c_10_reg_re, c_11_reg_re, c_12_reg_re, c_13_reg_re,
      c_20_reg_re, c_21_reg_re, c_22_reg_re, c_23_reg_re, c_30_reg_re, c_31_reg_re, c_32_reg_re,
      c_33_reg_re;
  


  uint64_t
      /* Point to the current elements in the four rows of A */
      *a_0p_pntr,
      *a_1p_pntr, *a_2p_pntr, *a_3p_pntr, *b_0p_pntr, *b_1p_pntr, *b_2p_pntr, *b_3p_pntr;
  a_0p_pntr = &A[0];
  
  a_1p_pntr = &A[n];
  
  a_2p_pntr = &A[2*n];
  a_3p_pntr = &A[3*n];
  
  b_0p_pntr = &B[0];
  b_1p_pntr = &B[n];
  b_2p_pntr = &B[2*n];
  b_3p_pntr = &B[3*n];
  c_00_reg_re = 0;
  c_01_reg_re = 0;
  c_02_reg_re = 0;
  c_03_reg_re = 0;

  c_10_reg_re = 0;
  c_11_reg_re = 0;
  c_12_reg_re = 0;
  c_13_reg_re = 0;

  c_20_reg_re = 0;
  c_21_reg_re = 0;
  c_22_reg_re = 0;
  c_23_reg_re = 0;

  c_30_reg_re = 0;
  c_31_reg_re = 0;
  c_32_reg_re = 0;
  c_33_reg_re = 0;
  
  for (p = 0; p + 4 <= n; p+=4) {
    __m256i a_0p_reg = _mm256_loadu_si256(a_0p_pntr);
    a_0p_pntr+=4;
    __m256i a_1p_reg = _mm256_loadu_si256(a_1p_pntr);
    a_1p_pntr+=4;
    __m256i a_2p_reg = _mm256_loadu_si256(a_2p_pntr);
    a_2p_pntr+=4;
    __m256i a_3p_reg = _mm256_loadu_si256(a_3p_pntr);
    a_3p_pntr+=4;
    __m256i b_0p_reg = _mm256_loadu_si256(b_0p_pntr);
    b_0p_pntr+=4;
    __m256i b_1p_reg = _mm256_loadu_si256(b_1p_pntr);
    b_1p_pntr+=4;
    __m256i b_2p_reg = _mm256_loadu_si256(b_2p_pntr);
    b_2p_pntr+=4;
    __m256i b_3p_reg = _mm256_loadu_si256(b_3p_pntr);
    b_3p_pntr+=4;
    c_00_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_0p_reg, b_0p_reg)), 7)<<1) - 256);
    c_01_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_0p_reg, b_1p_reg)), 7)<<1) - 256);
    c_02_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_0p_reg, b_2p_reg)), 7)<<1) - 256);
    c_03_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_0p_reg, b_3p_reg)), 7)<<1) - 256);

    c_10_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_1p_reg, b_0p_reg)), 7)<<1) - 256);
    c_11_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_1p_reg, b_1p_reg)), 7)<<1) - 256);
    c_12_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_1p_reg, b_2p_reg)), 7)<<1) - 256);
    c_13_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_1p_reg, b_3p_reg)), 7)<<1) - 256);

    c_20_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_2p_reg, b_0p_reg)), 7)<<1) - 256);
    c_21_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_2p_reg, b_1p_reg)), 7)<<1) - 256);
    c_22_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_2p_reg, b_2p_reg)), 7)<<1) - 256);
    c_23_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_2p_reg, b_3p_reg)), 7)<<1) - 256);

    c_30_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_3p_reg, b_0p_reg)), 7)<<1) - 256);
    c_31_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_3p_reg, b_1p_reg)), 7)<<1) - 256);
    c_32_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_3p_reg, b_2p_reg)), 7)<<1) - 256);
    c_33_reg_re -= ((_mm256_extract_epi32(popcnt256(_mm256_xor_si256(a_3p_reg, b_3p_reg)), 7)<<1) - 256);



  }
  if(n % 4 !=0){
    //printf("%d %d", p, n);
    for(;p < n; p++){
    a_0p_reg_re = *a_0p_pntr++;
    a_1p_reg_re = *a_1p_pntr++;
    a_2p_reg_re = *a_2p_pntr++;
    a_3p_reg_re = *a_3p_pntr++;

    b_p0_reg_re = *b_0p_pntr++;
    b_p1_reg_re = *b_1p_pntr++;
    b_p2_reg_re = *b_2p_pntr++;
    b_p3_reg_re = *b_3p_pntr++;

    /* First row */
    c_00_reg_re -= (popcnt64(MASK(a_0p_reg_re^b_p0_reg_re))<<1);
    c_01_reg_re -= (popcnt64(MASK(a_0p_reg_re^b_p1_reg_re))<<1);
    c_02_reg_re -= (popcnt64(MASK(a_0p_reg_re^b_p2_reg_re))<<1);
    c_03_reg_re -= (popcnt64(MASK(a_0p_reg_re^b_p3_reg_re))<<1);


    /* Second row */
    c_10_reg_re -= (popcnt64(MASK(a_1p_reg_re^b_p0_reg_re))<<1);
    c_11_reg_re -= (popcnt64(MASK(a_1p_reg_re^b_p1_reg_re))<<1);
    c_12_reg_re -= (popcnt64(MASK(a_1p_reg_re^b_p2_reg_re))<<1);
    c_13_reg_re -= (popcnt64(MASK(a_1p_reg_re^b_p3_reg_re))<<1);


    /* Third row */
    c_20_reg_re -= (popcnt64(MASK(a_2p_reg_re^b_p0_reg_re))<<1);
    c_21_reg_re -= (popcnt64(MASK(a_2p_reg_re^b_p1_reg_re))<<1);
    c_22_reg_re -= (popcnt64(MASK(a_2p_reg_re^b_p2_reg_re))<<1);
    c_23_reg_re -= (popcnt64(MASK(a_2p_reg_re^b_p3_reg_re))<<1);


    /* Four row */
    c_30_reg_re -= (popcnt64(MASK(a_3p_reg_re^b_p0_reg_re))<<1);
    c_31_reg_re -= (popcnt64(MASK(a_3p_reg_re^b_p1_reg_re))<<1);
    c_32_reg_re -= (popcnt64(MASK(a_3p_reg_re^b_p2_reg_re))<<1);
    c_33_reg_re -= (popcnt64(MASK(a_3p_reg_re^b_p3_reg_re))<<1);
    }
  }
  C[0] += c_00_reg_re;
  C[1] += c_01_reg_re;
  C[2] += c_02_reg_re;
  C[3] += c_03_reg_re;
  C[col] += c_10_reg_re;
  C[col + 1] += c_11_reg_re;
  C[col + 2] += c_12_reg_re;
  C[col + 3] += c_13_reg_re;
  C[2*col] += c_20_reg_re;
  C[2*col + 1] += c_21_reg_re;
  C[2*col + 2] += c_22_reg_re;
  C[2*col + 3] += c_23_reg_re;
  C[3*col] += c_30_reg_re;
  C[3*col + 1] += c_31_reg_re;
  C[3*col + 2] += c_32_reg_re;
  C[3*col + 3]+= c_33_reg_re;
}
void AddDot4x4(uint64_t *A, uint64_t *B, float *C, int m, int n, int col){
  int p;
  
  register uint64_t

      c_00_reg,
      c_01_reg, c_02_reg, c_03_reg, c_10_reg, c_11_reg, c_12_reg, c_13_reg,
      c_20_reg, c_21_reg, c_22_reg, c_23_reg, c_30_reg, c_31_reg, c_32_reg,
      c_33_reg,
      a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_p0_reg, b_p1_reg, b_p2_reg,
      b_p3_reg;

  uint64_t
      /* Point to the current elements in the four rows of A */
      *a_0p_pntr,
      *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

  a_0p_pntr = &A[0];
  a_1p_pntr = &A[n];
  a_2p_pntr = &A[2*n];
  a_3p_pntr = &A[3*n];

  c_00_reg = 0;
  c_01_reg = 0;
  c_02_reg = 0;
  c_03_reg = 0;
  c_10_reg = 0;
  c_11_reg = 0;
  c_12_reg = 0;
  c_13_reg = 0;
  c_20_reg = 0;
  c_21_reg = 0;
  c_22_reg = 0;
  c_23_reg = 0;
  c_30_reg = 0;
  c_31_reg = 0;
  c_32_reg = 0;
  c_33_reg = 0;

  for (p = 0; p < n; p++) {
    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;

    b_p0_reg = B[p];
    b_p1_reg = B[n + p];
    b_p2_reg = B[2 * n + p];
    b_p3_reg = B[3 * n + p];

    /* First row */
    //c_00_reg += a_0p_reg * b_p0_reg;
    c_00_reg -= (popcnt64(MASK(a_0p_reg^b_p0_reg))<<1);
    c_01_reg -= (popcnt64(MASK(a_0p_reg^b_p1_reg))<<1);
    c_02_reg -= (popcnt64(MASK(a_0p_reg^b_p2_reg))<<1);
    c_03_reg -= (popcnt64(MASK(a_0p_reg^b_p3_reg))<<1);
    //c_01_reg += a_0p_reg * b_p1_reg;
    //c_02_reg += a_0p_reg * b_p2_reg;
    //c_03_reg += a_0p_reg * b_p3_reg;

    /* Second row */
    c_10_reg -= (popcnt64(MASK(a_1p_reg^b_p0_reg))<<1);
    c_11_reg -= (popcnt64(MASK(a_1p_reg^b_p1_reg))<<1);
    c_12_reg -= (popcnt64(MASK(a_1p_reg^b_p2_reg))<<1);
    c_13_reg -= (popcnt64(MASK(a_1p_reg^b_p3_reg))<<1);
    //c_10_reg += a_1p_reg * b_p0_reg;
    //c_11_reg += a_1p_reg * b_p1_reg;
    //c_12_reg += a_1p_reg * b_p2_reg;
    //c_13_reg += a_1p_reg * b_p3_reg;

    /* Third row */
    c_20_reg -= (popcnt64(MASK(a_2p_reg^b_p0_reg))<<1);
    c_21_reg -= (popcnt64(MASK(a_2p_reg^b_p1_reg))<<1);
    c_22_reg -= (popcnt64(MASK(a_2p_reg^b_p2_reg))<<1);
    c_23_reg -= (popcnt64(MASK(a_2p_reg^b_p3_reg))<<1);
    //c_20_reg += a_2p_reg * b_p0_reg;
    //c_21_reg += a_2p_reg * b_p1_reg;
    //c_22_reg += a_2p_reg * b_p2_reg;
    //c_23_reg += a_2p_reg * b_p3_reg;

    /* Four row */
    c_30_reg -= (popcnt64(MASK(a_3p_reg^b_p0_reg))<<1);
    c_31_reg -= (popcnt64(MASK(a_3p_reg^b_p1_reg))<<1);
    c_32_reg -= (popcnt64(MASK(a_3p_reg^b_p2_reg))<<1);
    c_33_reg -= (popcnt64(MASK(a_3p_reg^b_p3_reg))<<1);
    //c_30_reg += a_3p_reg * b_p0_reg;
    //c_31_reg += a_3p_reg * b_p1_reg;
    //c_32_reg += a_3p_reg * b_p2_reg;
    //c_33_reg += a_3p_reg * b_p3_reg;
  }

  C[0] += c_00_reg;
  C[1] += c_01_reg;
  C[2] += c_02_reg;
  C[3] += c_03_reg;
  C[col] += c_10_reg;
  C[col + 1] += c_11_reg;
  C[col + 2] += c_12_reg;
  C[col + 3] += c_13_reg;
  C[2*col] += c_20_reg;
  C[2*col + 1] += c_21_reg;
  C[2*col + 2] += c_22_reg;
  C[2*col + 3] += c_23_reg;
  C[3*col] += c_30_reg;
  C[3*col + 1] += c_31_reg;
  C[3*col + 2] += c_32_reg;
  C[3*col + 3]+= c_33_reg;
}
void normal_multiply_ijk(uint64_t *A, uint64_t *B, float *C, int m, int n, int col) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < col; j++) {
            for (int k = 0; k < n; k++) {
                //C[i * col + j] += A[i * n + k] * B[j * n + k];
                C[i * col + j] -= (popcnt64(MASK(A[i * n + k]^B[j * n + k]))<<1);
                //C[i * col + j] -= ((popcnt64(A[i * n + k]^B[j * n + k])<<1) - 64);
            }
        }
    }
}
void
dgemm_nn_ours(int            m,
         int            n,
         int            kk,
         uint64_t      *A,
         int            incRowA,
         int            incColA,
         uint64_t      *B,
         int            incRowB,
         int            incColB,
         float         *C,
         int            incRowC,
         int            incColC,
         int            beta,
         int            alpha,
         int         alphas)
{   //printf("%d %d %d\n" ,m,n,incRowA);
    int block_size = 32;
    for(int i =0; i<m; i+= block_size){
        for(int j = 0;j<n; j+=block_size){
            for(int k = 0; k < incRowA; k+=block_size){
                for(int ii = i; ii < i + block_size && ii < m; ii++){
                    for(int jj = j; jj< j + block_size && jj < n; jj++){
                        int result = 0;
                        for(int kk_ = k; kk_ < k + block_size&& kk_ < incRowA; kk_++){
                            result -= popcnt64(MASK(A[ii * incRowA + kk_]^B[jj * incRowA + kk_]))<<1;
                        }
                        C[ii * n + jj] += result;
                    }
                }
            }
            //C[i * n + j] = result;
        }
    }
}
void binary_gemm_cpu_4_activation(THLongTensor* a1, THLongTensor* a2, THLongTensor* a3, THLongTensor* a4, THLongTensor* b1, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha, THFloatTensor* alphas){
    //clock_t start_time, end_time;
    //double cpu_time_used,total = 0;    
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }
    uint64_t *A1 = (uint64_t*)THLongTensor_data(a1);
    uint64_t *B1 = (uint64_t*)THLongTensor_data(b1);
    uint64_t *A2 = (uint64_t*)THLongTensor_data(a2);
    //uint64_t *B2 = (uint64_t*)THLongTensor_data(b2);
    uint64_t *A3 = (uint64_t*)THLongTensor_data(a3);
    //uint64_t *B3 = (uint64_t*)THLongTensor_data(b3);
    uint64_t *A4 = (uint64_t*)THLongTensor_data(a4);
    //uint64_t *B4 = (uint64_t*)THLongTensor_data(b4);    
        
    float *C = THFloatTensor_data(c);
    float *D = THFloatTensor_data(alphas);
    int n = 1 + (nn-1) / 64;
    //int n1 = 1 + (n-1) / 4;
    
    //__m256i A_1[m][n1], B_1[k][n1]; 
    //__m256i* A_1 = (__m256i*)_mm_malloc(m * n1 * sizeof(__m256i), 32);
    //__m256i* B_1 = (__m256i*)_mm_malloc(k * n1 * sizeof(__m256i), 32);
    //__m256i* A_2 = (__m256i*)_mm_malloc(m * n1 * sizeof(__m256i), 32);
    //__m256i* B_2 = (__m256i*)_mm_malloc(k * n1 * sizeof(__m256i), 32);
    //__m256i* A_3 = (__m256i*)_mm_malloc(m * n1 * sizeof(__m256i), 32);
    //__m256i* B_3 = (__m256i*)_mm_malloc(k * n1 * sizeof(__m256i), 32);
    //__m256i* A_4 = (__m256i*)_mm_malloc(m * n1 * sizeof(__m256i), 32);
    //__m256i* B_4 = (__m256i*)_mm_malloc(k * n1 * sizeof(__m256i), 32);

    
    //encode_rows_cpu_kernel_256(A1, A_1, m, n);

    //encode_rows_cpu_kernel_256(B1, B_1, k, n);
    //encode_rows_cpu_kernel_256(A2, A_2, m, n);

    //encode_rows_cpu_kernel_256(B2, B_2, k, n);
    //encode_rows_cpu_kernel_256(A3, A_3, m, n);

    //encode_rows_cpu_kernel_256(B3, B_3, k, n);
    //encode_rows_cpu_kernel_256(A4, A_4, m, n);

    //encode_rows_cpu_kernel_256(B4, B_4, k, n);
    //exit(0);
    //size_t arraySize = sizeof(A_1) / sizeof(A_1[0]);
    //
    //exit(0);
    //int n = 1 + (nn-1) / 4;
    int brow = transb? 1:k, bcol = transb? n:1;
    //start_time = clock();
    dgemm_nn_ours_4_activation(m, k, nn, A1, A2, A3, A4, n, 1, B1, brow, bcol, C, k, 1, beta, alpha, D);
    //end_time = clock();
    //cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    //free(A1);
    //free(B1);
    //free(A2);
    //free(B_2);
    //free(A3);
    //free(B_3);
    //free(A4);
    //free(B_4);
}
void
dgemm_nn_ours_4_activation(int            m,
         int            n,
         int            kk_,
         uint64_t      *A1,
         uint64_t      *A2,
         uint64_t      *A3,
         uint64_t      *A4,
         int            incRowA,
         int            incColA,
         uint64_t      *B1,
         int            incRowB,
         int            incColB,
         float         *C,
         int            incRowC,
         int            incColC,
         int            beta,
         int            alpha,
         float         *alphas)
{   //printf("%d %d %d\n" ,m,n,incRowA);

    int temp1, temp2, temp3, temp4, block_size = 4;
    for(int i = 0 ; i < m; i++){
        for(int j = 0; j < n; j++){
            int result = 0;
            for(int k = 0; k < incRowA; k++){
                //temp1 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A1[i * incRowA + k], B1[j * incRowA + k])), 7);
                //temp2 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A2[i * incRowA + k], B1[j * incRowA + k])), 7);
                //temp3 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A3[i * incRowA + k], B1[j * incRowA + k])), 7);
                //temp4 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A4[i * incRowA + k], B1[j * incRowA + k])), 7);
                temp1 = popcnt64(A1[i * incRowA + k]^B1[j * incRowA + k]);
                temp2 = popcnt64(A2[i * incRowA + k]^B1[j * incRowA + k]);
                temp3 = popcnt64(A3[i * incRowA + k]^B1[j * incRowA + k]);
                temp4 = popcnt64(A4[i * incRowA + k]^B1[j * incRowA + k]);
                result -= ((temp1<<1) -64);
                result -= ((temp2<<2) - 128);
                result -= ((temp3<<3) - 256);
                result -= ((temp4<<4) - 512);                  
            }
            C[i * n + j] = result;
        }
    }
    /*
    for(int i =0; i<m; i+= block_size){
        for(int j = 0;j<n; j+=block_size){
            for(int k = 0; k < incRowA; k+=block_size){
                for(int ii = i; ii < i + block_size && ii < m; ii++){
                    for(int jj = j; jj < j + block_size && jj < n; jj++){
                        int result = 0;
                        for(int kk = k; kk < k + block_size&& kk < incRowA; kk++){
                            temp1 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A1[ii * incRowA + kk], B1[jj * incRowA + kk])), 7);
                            temp2 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A2[ii * incRowA + kk], B1[jj * incRowA + kk])), 7);
                            temp3 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A3[ii * incRowA + kk], B1[jj * incRowA + kk])), 7);
                            temp4 = _mm256_extract_epi32(popcnt256(_mm256_xor_si256(A4[ii * incRowA + kk], B1[jj * incRowA + kk])), 7);
                            result -= ((temp1<<1) -256);
                            result -= ((temp2<<2) - 256);
                            result -= ((temp3<<3) - 256);
                            result -= ((temp4<<4) - 256);                            
                        }
                        C[ii * n + jj] += result;
                    }
                }
                //result -= popcnt64(MASK(A[i * incRowA + k]^B[j * incRowA + k]))<<1;
                //start_time = clock();

                //end_time = clock();
                //cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
                //total += cpu_time_used;
                
            }
            
        }
    }*/

}
void THNN_unfolded_copy(
                        THFloatTensor *columns,
                        THFloatTensor *input,
                        int kW, int kH,
                        int dW, int dH,
                        int padW, int padH,
                        int nInputPlane,
                        int inputWidth, int inputHeight,
                        int outputWidth, int outputHeight)
{
    // This function assumes that
    // kH*kW does not overflow an int
    // nInputPlane*kH*kW does not overflow a int64_t
    // outputHeight*dH does not overflow a int64_t
    // outputWidth*dW does not overflow a int64_t

    int64_t k;
    float *input_data = THFloatTensor_data(input);
    float *columns_data = THFloatTensor_data(columns);

#pragma omp parallel for private(k)
    for(k = 0; k < (int64_t)nInputPlane*kH*kW; k++) {
        int64_t nip = k / (kH*kW);
        int64_t rest = k % (kH*kW);
        int64_t kh = rest / kW;
        int64_t kw = rest % kW;
        int x, y;
        int64_t ix, iy;
        float *dst = columns_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
        float *src = input_data + nip*((size_t)inputHeight*inputWidth);
        if (padW > 0 || padH > 0) {
            int64_t lpad,rpad;
            for(y = 0; y < outputHeight; y++) {
                iy = (int64_t)y*dH - padH + kh;
                if (iy < 0 || iy >= inputHeight) {
                    memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*outputWidth);
                } else {
                    if (dW==1){
                        ix = 0 - padW + kw;
                        lpad = fmaxf(0,padW-kw);
                        rpad = fmaxf(0,padW-(kW-kw-1));
                        if (outputWidth-rpad-lpad <= 0) {
                            memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*outputWidth);
                        } else {
                            if (lpad > 0) memset(dst+(size_t)y*outputWidth, 0, sizeof(float)*lpad);
                            memcpy(dst+(size_t)y*outputWidth+lpad, src+(size_t)iy*inputWidth+ix+lpad, sizeof(float)*(outputWidth-rpad-lpad));
                            if (rpad > 0) memset(dst+(size_t)y*outputWidth + outputWidth - rpad, 0, sizeof(float)*rpad);
                        }
                    }
                    else{
                        for (x=0; x<outputWidth; x++){
                            ix = (int64_t)x*dW - padW + kw;
                            if (ix < 0 || ix >= inputWidth)
                                memset(dst+(size_t)y*outputWidth+x, 0, sizeof(float)*1);
                            else
                                memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix, sizeof(float)*(1));
                        }
                    }
                }
            }
        } else {
            for(y = 0; y < outputHeight; y++) {
                iy = (int64_t)y*dH + kh;
                ix = 0 + kw;
                if (dW == 1)
                    memcpy(dst+(size_t)y*outputWidth, src+(size_t)iy*inputWidth+ix, sizeof(float)*outputWidth);
                else{
                    for (x=0; x<outputWidth; x++)
                        memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix+(int64_t)x*dW, sizeof(float)*(1));
                }
            }
        }
    }
}

static void THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
                                                             THFloatTensor *output,
                                                             THIntTensor *weight,
                                                             THFloatTensor *bias,
                                                             THFloatTensor *ones,
                                                             THIntTensor *bin_col,
                                                             THFloatTensor *alphas,
                                                             int kW, int kH,
                                                             int dW, int dH,
                                                             int padW, int padH,
                                                             int64_t nInputPlane,
                                                             int64_t inputWidth, int64_t inputHeight,
                                                             int64_t nOutputPlane,
                                                             int64_t outputWidth, int64_t outputHeight)
{
    THFloatTensor *output2d;

    output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset, nOutputPlane, -1, outputHeight*outputWidth, -1);
    THFloatTensor_zero(output2d);

    binary_gemm_cpu(weight, bin_col, output2d, nOutputPlane, kW*kH*nInputPlane, outputHeight*outputWidth, 0, 1, 1, alphas);
    if (bias->nDimension) {
        THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
    }
    THFloatTensor_free(output2d);
}

void THNN_Bin_SpatialConvolutionMM_updateOutput(
                                                THFloatTensor *input,
                                                THFloatTensor *output,
                                                THIntTensor *weight,
                                                THFloatTensor *bias,
                                                THFloatTensor *columns,
                                                THFloatTensor *alphas,
                                                int kH, int kW,
                                                int dH, int dW,
                                                int padH, int padW)
{
    THIntTensor *bin_col = THIntTensor_new();
    THFloatTensor *ones  = THFloatTensor_new();
    input = THFloatTensor_newContiguous(input);
    int ndim = input->nDimension;
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;
//    clock_t start_time, end_time;
//    double cpu_time_used;


    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    int64_t nInputPlane  = input->size[dimf];
    int64_t inputHeight  = input->size[dimh];
    int64_t inputWidth   = input->size[dimw];
    int64_t nOutputPlane = weight->size[0];
    int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
    int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

    if (bias->nDimension ==1) {
        THFloatTensor_resize2d(bias, bias->size[0], 1);
    }


    THFloatTensor_resize2d(ones, 1, outputHeight*outputWidth);
    THFloatTensor_fill(ones, 1);

    int64_t T = input->size[0];
    int64_t t;

    THFloatTensor_resize4d(output, T, nOutputPlane, outputHeight, outputWidth);
    THFloatTensor_resize3d(columns, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THIntTensor_resize3d(bin_col, T, weight->size[0], outputHeight*outputWidth);
//#pragma omp parallel for private(t)


//    
    for(t = 0; t < T; t++)
    {
        THFloatTensor *input_t = THFloatTensor_newSelect(input, 0, t);
        THFloatTensor *columns_t = THFloatTensor_newSelect(columns, 0, t);
        THIntTensor *bin_col_t = THIntTensor_newSelect(bin_col, 0, t);

        THNN_unfolded_copy(
            columns_t, input_t, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight
        );
        encode_cols_cpu(columns_t, bin_col_t);

        THFloatTensor_free(input_t);
        THFloatTensor_free(columns_t);
        THIntTensor_free(bin_col_t);
    }



    for(t = 0; t < T; t++){
        THFloatTensor *output_t = THFloatTensor_newSelect(output, 0, t);
        THIntTensor *bin_col_t = THIntTensor_newSelect(bin_col, 0, t);

        THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
            output_t, weight, bias, ones, bin_col_t, alphas, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight
        );

        THFloatTensor_free(output_t);
        THIntTensor_free(bin_col_t);
    }
    THFloatTensor_free(input);
    THFloatTensor_free(ones);
    THIntTensor_free(bin_col);
}
