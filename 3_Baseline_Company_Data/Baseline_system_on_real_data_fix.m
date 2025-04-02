%% Clear workspace and close figures
clear all;
close all;
clc;

%% Define the path for the data
[primary, fs] = audioread("C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Data from AI heathway\Data_ANC\Experiment_Data\Hospital Ambient Noises\NHS\1\primary.wav");   %noise + clean signal
[noise, ~] = audioread("C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Data from AI heathway\Data_ANC\Experiment_Data\Hospital Ambient Noises\NHS\1\secondary.wav");
[clean, ~] = audioread("C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Data from AI heathway\Data_ANC\Experiment_Data\Hospital Ambient Noises\NHS\1\ZCH0019.wav");

% Find the minimum length
minLength = min([length(primary), length(noise), length(clean)]);

% Truncate signals to the same length
primary = primary(1:minLength);
noise = noise(1:minLength);
clean = clean(1:minLength);

%% Calculate initial SNR for audio files
initial_SNR = 10 * log10(sum(clean.^2) / sum((clean - noise).^2));

fprintf('Initial SNR (before any processing): %.2f dB\n', initial_SNR);

%% Fix #1: Bandpass Filtering to Remove Distortions
lowCutoff = 5; % Low cut frequency in Hz
highCutoff = 800; % High cut frequency in Hz
[b, a] = butter(4, [lowCutoff, highCutoff] / (fs/2), 'bandpass');
primary = filtfilt(b, a, primary);
noise = filtfilt(b, a, noise);

% SNR after fix #1 (Bandpass Filtering)
SNR_after_bandpass = 10 * log10(sum(clean.^2) / sum((clean - noise).^2));
fprintf('SNR after Bandpass Filtering (Fix #1): %.2f dB\n', SNR_after_bandpass);

%% Fix #2: Align Reference Noise Using Cross-Correlation
[xcorr_vals, lags] = xcorr(primary, noise);
[~, maxIdx] = max(abs(xcorr_vals));
delay = lags(maxIdx);

if delay > 0
    noise = [zeros(delay,1); noise(1:end-delay)];
elseif delay < 0
    noise = noise(-delay+1:end);
    noise = [noise; zeros(-delay,1)];
end

% SNR after fix #2 (Cross-Correlation Delay Alignment)
SNR_after_alignment = 10 * log10(sum(clean.^2) / sum((clean - noise).^2));
fprintf('SNR after Noise Alignment (Fix #2): %.2f dB\n', SNR_after_alignment);

%% Fix #3: Hilbert Transform for Phase Correction
analytic_primary = hilbert(primary);
analytic_noise = hilbert(noise);
phase_diff = angle(analytic_primary ./ analytic_noise); % Compute phase shift per sample

% Apply phase correction to noise
noise = real(noise .* exp(-1j * phase_diff));

% SNR after fix #3 (Hilbert Transform Phase Correction)
SNR_after_phase_correction = 10 * log10(sum(clean.^2) / sum((clean - noise).^2));
fprintf('SNR after Phase Correction (Fix #3): %.2f dB\n', SNR_after_phase_correction);
%% LMS Filter
M = 12; 
mu_LMS = 0.1; 
w_LMS = zeros(M, 1); 
padded_signal = [zeros(M-1, 1); primary]; 
output_LMS = zeros(minLength, 1);

for n = 1:minLength
    u_vect = padded_signal(n:n+M-1); 
    e = noise(n) - w_LMS' * u_vect; 
    w_LMS = w_LMS + mu_LMS * e * u_vect; 
    output_LMS(n) = w_LMS' * u_vect;
end

filtered_signal_LMS = primary - output_LMS; 

%% NLMS Filter
mu_NLMS = 1; 
w_NLMS = zeros(M, 1); 
output_NLMS = zeros(minLength, 1);
Eps = 0.0001; 

for n = 1:minLength
    u_vect = padded_signal(n:n+M-1);
    mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2);
    e = noise(n) - w_NLMS' * u_vect;
    w_NLMS = w_NLMS + mu_adapt * e * u_vect;
    output_NLMS(n) = w_NLMS' * u_vect;
end

filtered_signal_NLMS = primary - output_NLMS; 

%% RLS Filter
lambda = 1; % Forgetting factor  1 - 1 / (0.1 * M) =  0.1666 ; 
delta = 0.01; 
P = 1 / delta * eye(M); 
w_RLS = zeros(M, 1);
output_RLS = zeros(minLength, 1);

for n = 1:minLength
    u_vect = padded_signal(n:n+M-1); 
    PI = P * u_vect; 
    gain_k = PI / (lambda + u_vect' * PI); 
    e = noise(n) - w_RLS' * u_vect; 
    w_RLS = w_RLS + gain_k * e; 
    P = P / lambda - gain_k * (u_vect' * P) / lambda; 
    output_RLS(n) = w_RLS' * u_vect;
end

filtered_signal_RLS = primary - output_RLS; 

%% Calculate SNR Improvement
filtered_SNR_LMS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_LMS).^2));
filtered_SNR_NLMS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_NLMS).^2));
filtered_SNR_RLS = 10 * log10(sum(clean.^2) / sum((clean - filtered_signal_RLS).^2));

fprintf('SNR after LMS Filter: %.2f dB\n', mean(filtered_SNR_LMS));
fprintf('SNR after NLMS Filter: %.2f dB\n', mean(filtered_SNR_NLMS));
fprintf('SNR after RLS Filter: %.2f dB\n', mean(filtered_SNR_RLS));

%% Visualization of Results
figure;
subplot(5, 1, 1);
plot(clean);
title('Clean Signal');
xlim([0 minLength]);

subplot(5, 1, 2);
plot(primary);
title('Noisy Signal');
xlim([0 minLength]);

subplot(5, 1, 3);
plot(filtered_signal_LMS);
title('Filtered Signal (LMS)');
xlim([0 minLength]);

subplot(5, 1, 4);
plot(filtered_signal_NLMS);
title('Filtered Signal (NLMS)');
xlim([0 minLength]);

subplot(5, 1, 5);
plot(filtered_signal_RLS);
title('Filtered Signal (RLS)');
xlim([0 minLength]);

%% Spectrogram
figure;
melSpectrogram(primary(1:16000*6),fs, ...
                   'Window',hann(256,'periodic'), ...
                   'OverlapLength',200, ...
                   'FFTLength',1024, ...
                   'NumBands',64, ...
                   'FrequencyRange',[62.5,8e3]);
colormap hot; colorbar;

% Save figure
tightfig();
saveas(gcf, 'Spectrogram.pdf');

% %% Cross-Correlation Before and After Noise Alignment
% figure;
% subplot(2,1,1);
% [xcorr_vals_before, lags_before] = xcorr(primary, noise);
% plot(lags_before/fs, xcorr_vals_before);
% title('Cross-Correlation Before Noise Alignment');
% xlabel('Time Lag (s)');
% ylabel('Cross-Correlation');
% 
% subplot(2,1,2);
% [xcorr_vals_after, lags_after] = xcorr(primary, noise);
% plot(lags_after/fs, xcorr_vals_after);
% title('Cross-Correlation After Noise Alignment');
% xlabel('Time Lag (s)');
% ylabel('Cross-Correlation');
% 
% saveas(gcf, 'Cross_Correlation_Alignment.pdf');

% %% Phase Difference Before and After Fix #3
% figure;
% subplot(2,1,1);
% plot(angle(analytic_primary ./ analytic_noise));
% title('Phase Difference Before Fix #3');
% xlabel('Sample Index');
% ylabel('Phase (radians)');
% 
% subplot(2,1,2);
% plot(phase_diff); % Now it shows the phase correction per sample
% title('Phase Difference After Fix #3');
% xlabel('Sample Index');
% ylabel('Phase (radians)');
% 
% saveas(gcf, 'Phase_Correction.pdf');

%% Export filtered signals
filtered_signal_LMS = filtered_signal_LMS / max(abs(filtered_signal_LMS));
filtered_signal_NLMS = filtered_signal_NLMS / max(abs(filtered_signal_NLMS));
filtered_signal_RLS = filtered_signal_RLS / max(abs(filtered_signal_RLS));

audiowrite('Filtered_LMS.wav', filtered_signal_LMS, fs);
audiowrite('Filtered_NLMS.wav', filtered_signal_NLMS, fs);
audiowrite('Filtered_RLS.wav', filtered_signal_RLS, fs);

fprintf('Filtered signals saved as audio files.\n\n');

% Tuning Parementers for tables
fprintf('Sampling rate: %f Hz\n', fs);
fprintf('Filter order (LMS, NLMS, RLS): %f \n', M);
fprintf('Step size (LMS): %f \n', mu_LMS);
fprintf('Step size (NLMS): %f \n', mu_NLMS);
fprintf('Forgetting factor (RLS): %f \n', lambda);

%% Save Figure as PDF
tightfig();
saveas(gcf, 'Hospital_Ambient_Noises_NHS_1.pdf');

function tightfig()
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end
