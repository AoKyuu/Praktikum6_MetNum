clc; clear; close all;

A = [ 5, -1, -1,  0,  0;
     -1,  5, -1, -1,  0;
     -1, -1,  4, -1, -1;
      0,  0,  1,  4, -2;
      0,  1, -1,  1,  4];

b = [-1; 2; 6; 2; -1];
n = length(b);

% Parameter metode Iteratif
max_iter = 20;
toleransi = 0.001;

fprintf('PENYELESAIAN SPL DAN WAKTU KOMPUTASI (1 dan 3)\n');

% GAUSS
tic; % Start timer Gauss
Aug = [A b];
x_gauss = zeros(n,1);
% Forward Elimination
for k = 1:n-1
    for i = k+1:n
        factor = Aug(i,k)/Aug(k,k);
        Aug(i,:) = Aug(i,:) - factor*Aug(k,:);
    end
end
% Backward Substitution
x_gauss(n) = Aug(n,n+1)/Aug(n,n);
for i = n-1:-1:1
    x_gauss(i) = (Aug(i,n+1) - Aug(i,i+1:n)*x_gauss(i+1:n))/Aug(i,i);
end
t_gauss = toc; % Stop timer Gauss

fprintf('1. Metode Eliminasi Gauss\n');
fprintf('   Waktu Komputasi : %.6f detik\n', t_gauss);
fprintf('   Solusi [p,q,r,s,t] : [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n', x_gauss);

% GAUSS-JORDAN
tic; % Start timer Gauss-Jordan
Aug_gj = [A b];
for k = 1:n
    Aug_gj(k,:) = Aug_gj(k,:) / Aug_gj(k,k); % Normalisasi baris pivot
    for i = 1:n
        if i ~= k
            factor = Aug_gj(i,k);
            Aug_gj(i,:) = Aug_gj(i,:) - factor*Aug_gj(k,:);
        end
    end
end
x_gj = Aug_gj(:,n+1);
t_gj = toc; % Stop timer Gauss-Jordan

fprintf('2. Metode Eliminasi Gauss-Jordan\n');
fprintf('   Waktu Komputasi : %.6f detik\n', t_gj);
fprintf('   Solusi [p,q,r,s,t] : [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n', x_gj);

% DEKOMPOSISI LU
tic; % Start timer Dekomposisi LU
[L, U, P] = lu(A);
y = L\(P*b);
x_lu = U\y;
t_lu = toc; % Stop timer Dekomposisi LU

fprintf('3. Metode Dekomposisi LU\n');
fprintf('   Waktu Komputasi : %.6f detik\n', t_lu);
fprintf('   Solusi [p,q,r,s,t] : [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n', x_lu);

% JACOBI
tic; % Start timer Jacobi
x_jac = zeros(n,1); % Tebakan awal
x_jac_new = zeros(n,1);
for iter_jac = 1:max_iter
    for i = 1:n
        sum_Ax = A(i,:)*x_jac - A(i,i)*x_jac(i);
        x_jac_new(i) = (b(i) - sum_Ax) / A(i,i);
    end
    % Cek toleransi
    if max(abs(x_jac_new - x_jac)) < toleransi
        break;
    end
    x_jac = x_jac_new;
end
x_jac = x_jac_new;
t_jacobi = toc; % Stop timer Jacobi

fprintf('4. Metode Jacobi (Berhenti pada iterasi ke-%d)\n', iter_jac);
fprintf('   Waktu Komputasi : %.6f detik\n', t_jacobi);
fprintf('   Solusi [p,q,r,s,t] : [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n', x_jac);

% GAUSS-SEIDEL
tic; % Start timer Gauss-Seidel
x_gs = zeros(n,1); % Tebakan awal
for iter_gs = 1:max_iter
    x_gs_old = x_gs;
    for i = 1:n
        sum1 = A(i,1:i-1) * x_gs(1:i-1);     % Gunakan nilai yang sudah diupdate (jika ada)
        sum2 = A(i,i+1:n) * x_gs_old(i+1:n); % Gunakan nilai lama
        x_gs(i) = (b(i) - sum1 - sum2) / A(i,i);
    end
    % Cek toleransi
    if max(abs(x_gs - x_gs_old)) < toleransi
        break;
    end
end
t_gs = toc; % Stop timer Gauss-Seidel

fprintf('5. Metode Gauss-Seidel (Berhenti pada iterasi ke-%d)\n', iter_gs);
fprintf('   Waktu Komputasi : %.6f detik\n', t_gs);
fprintf('   Solusi [p,q,r,s,t] : [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n', x_gs);


%% PERHITUNGAN GALAT (ERROR)

fprintf('GALAT METODE ITERATIF TERHADAP METODE LANGSUNG (2)\n');

% Saya memakai Metode Dekomposisi LU sebagai "Nilai Eksak" metode analitik
nilai_eksak = x_lu;

% Menghitung galat abstrak = abs(nilai eksak - nilai hampiran)
galat_jacobi = abs(nilai_eksak - x_jac);
galat_gs = abs(nilai_eksak - x_gs);

fprintf('Galat (Error) Abstrak [p, q, r, s, t]:\n');
fprintf('1. Galat Metode Jacobi       : [%.6f, %.6f, %.6f, %.6f, %.6f]\n', galat_jacobi);
fprintf('2. Galat Metode Gauss-Seidel : [%.6f, %.6f, %.6f, %.6f, %.6f]\n\n', galat_gs);

% Menampilkan rangkuman total (Max Error)
fprintf('Maksimum error Jacobi       : %.6f\n', max(galat_jacobi));
fprintf('Maksimum error Gauss-Seidel : %.6f\n', max(galat_gs));
