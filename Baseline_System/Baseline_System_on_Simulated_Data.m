% clear all;
% close all;
% clc;
% 
% % *************************************************************************
% % Simulate Signal and Noise
% fs = 8000; % Sampling frequency
% t = 0:1/fs:2; % Time vector (2 seconds)
% signal = sin(2 * pi * 10 * t)'; % Simulated clean signal (10 Hz sine wave)
% noise = 0.5 * randn(size(t))'; % Simulated Gaussian noise
% noisy_signal = signal + noise; % Combine signal and noise
% 
% % Calculate initial SNR
% initial_SNR = 10 * log10(sum(signal.^2) / sum(noise.^2));
% fprintf('Initial SNR: %.2f dB\n', initial_SNR);
% 
% % Filter Parameters
% N = length(noisy_signal); % Signal length
% M = 16; % Filter order
% 
% % *************************************************************************
% % LMS Filter
% mu_LMS = 0.01; % Step size for LMS
% w_LMS = zeros(M, 1); % Initialize filter weights
% padded_signal = [zeros(M-1, 1); noisy_signal]; % Pad noisy signal
% output_LMS = zeros(N, 1); % Filter output
% 
% for n = 1:N
%     u_vect = padded_signal(n:n+M-1); % Current input vector
%     e = signal(n) - w_LMS' * u_vect; % Error signal
%     w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
%     output_LMS(n) = w_LMS' * u_vect; % Filtered output
% end
% 
% % *************************************************************************
% % NLMS Filter
% mu_NLMS = 1; % Step size for NLMS
% w_NLMS = zeros(M, 1); % Initialize filter weights
% output_NLMS = zeros(N, 1); % Filter output
% Eps = 1e-6; % Stability constant
% 
% for n = 1:N
%     u_vect = padded_signal(n:n+M-1); % Current input vector
%     mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
%     e = signal(n) - w_NLMS' * u_vect; % Error signal
%     w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
%     output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
% end
% 
% % *************************************************************************
% % RLS Filter
% lambda = 0.99; % Forgetting factor
% delta = 1e-2; % Initialization constant
% P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
% w_RLS = zeros(M, 1); % Initialize filter weights
% output_RLS = zeros(N, 1); % Filter output
% 
% for n = 1:N
%     u_vect = padded_signal(n:n+M-1); % Current input vector
%     gain_k = P * u_vect / (lambda + u_vect' * P * u_vect); % Gain vector
%     e = signal(n) - w_RLS' * u_vect; % Error signal
%     w_RLS = w_RLS + gain_k * e; % Update weights
%     P = (P - gain_k * u_vect' * P) / lambda; % Update P matrix
%     output_RLS(n) = w_RLS' * u_vect; % Filtered output
% end
% 
% % *************************************************************************
% % Calculate SNR Improvement
% filtered_SNR_LMS = 10 * log10(sum(signal.^2) / sum((signal - output_LMS).^2));
% filtered_SNR_NLMS = 10 * log10(sum(signal.^2) / sum((signal - output_NLMS).^2));
% filtered_SNR_RLS = 10 * log10(sum(signal.^2) / sum((signal - output_RLS).^2));
% 
% fprintf('SNR after LMS Filter: %.2f dB\n', filtered_SNR_LMS);
% fprintf('SNR after NLMS Filter: %.2f dB\n', filtered_SNR_NLMS);
% fprintf('SNR after RLS Filter: %.2f dB\n', filtered_SNR_RLS);
% 
% % *************************************************************************
% % Visualization of Results
% figure;
% subplot(5, 1, 1);
% plot(signal);
% title('Clean Signal');
% xlabel('Sample Number');
% ylabel('Amplitude');
% xlim([0 16000]);
% 
% subplot(5, 1, 2);
% plot(noisy_signal);
% title('Noisy Signal');
% xlabel('Sample Number');
% ylabel('Amplitude');
% xlim([0 16000]);
% 
% subplot(5, 1, 3);
% plot(output_LMS);
% title('Filtered Signal (LMS)');
% xlabel('Sample Number');
% ylabel('Amplitude');
% xlim([0 16000]);
% 
% subplot(5, 1, 4);
% plot(output_NLMS);
% title('Filtered Signal (NLMS)');
% xlabel('Sample Number');
% ylabel('Amplitude');
% xlim([0 16000]);
% 
% subplot(5, 1, 5);
% plot(output_RLS);
% title('Filtered Signal (RLS)');
% xlabel('Sample Number');
% ylabel('Amplitude');
% xlim([0 16000]);
% 
% % *************************************************************************
% % Normalize filtered signals
% output_LMS = output_LMS / max(abs(output_LMS));
% output_NLMS = output_NLMS / max(abs(output_NLMS));
% output_RLS = output_RLS / max(abs(output_RLS));
% 
% % Save Filtered Signals
% audiowrite('Filtered_LMS.wav', output_LMS, fs);
% audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
% audiowrite('Filtered_RLS.wav', output_RLS, fs);
% 
% fprintf('Filtered signals saved as audio files.\n');
% 
% % Crop the figure and save as PDF
% tightfig();
% saveas(gcf, 'filtered_signals_comparison.pdf');
% 

%%
clear all;
close all;
clc;

% Simulate Signal and Noise
fs = 8000; % Sampling frequency
duration = 15; % Duration of the signal (in seconds)
t = 0:1/fs:duration; % Time vector
signal = sin(2 * pi * 10 * t)'; % Simulated clean signal (10 Hz sine wave)

% Generate noise at -20 dB SNR
signal_power = mean(signal.^2);
SNR_dB = -20; % Desired SNR in dB
SNR_linear = 10^(SNR_dB/10);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power) * randn(size(t))';

noisy_signal = signal + noise; % Combine signal and noise

% Calculate initial SNR
initial_SNR = 10 * log10(sum(signal.^2) / sum(noise.^2));
fprintf('Initial SNR: %.2f dB\n', initial_SNR);

% Filter Parameters
N = length(noisy_signal); % Signal length
M = 12; % Filter order (adjusted to match first code)

% Pad noisy signal
padded_signal = [zeros(M-1, 1); noisy_signal]; 

% LMS Filter
mu_LMS = 0.0006; % Step size for LMS (adjusted from first code)
w_LMS = zeros(M, 1); % Initialize filter weights
output_LMS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    e = signal(n) - w_LMS' * u_vect; % Error signal
    w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
    output_LMS(n) = w_LMS' * u_vect; % Filtered output
end 

% NLMS Filter
mu_NLMS = 1; % Step size for NLMS (kept from first code)
w_NLMS = zeros(M, 1); % Initialize filter weights
output_NLMS = zeros(N, 1); % Filter output
Eps = 0.0001; % Stability constant (adjusted from first code)

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
    e = signal(n) - w_NLMS' * u_vect; % Error signal
    w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
    output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
end

% RLS Filter
lambda = 1 - 1 / (0.1 * M); % Forgetting factor (adjusted from first code)
delta = 0.01; % Initialization constant (adjusted from first code)
P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
w_RLS = zeros(M, 1); % Initialize filter weights
padded_signal = [sqrt(delta) * randn(M-1, 1); noisy_signal]; % Pad noisy signal
output_RLS = zeros(N, 1); % Filter output

for n = 1:N
    u_vect = padded_signal(n:n+M-1); % Current input vector
    PI = P * u_vect; % Intermediate calculation
    gain_k = PI / (lambda + u_vect' * PI); % Gain
    e = signal(n) - w_RLS' * u_vect; % Error signal
    w_RLS = w_RLS + gain_k * e; % Update weights
    P = P / lambda - gain_k * (u_vect' * P) / lambda; % Update P matrix
    output_RLS(n) = w_RLS' * u_vect; % Filtered output
end

% Calculate SNR Improvement
filtered_SNR_LMS = 10 * log10(sum(signal.^2) / sum((signal - output_LMS).^2));
filtered_SNR_NLMS = 10 * log10(sum(signal.^2) / sum((signal - output_NLMS).^2));
filtered_SNR_RLS = 10 * log10(sum(signal.^2) / sum((signal - output_RLS).^2));

fprintf('SNR after LMS Filter: %.2f dB\n', filtered_SNR_LMS);
fprintf('SNR after NLMS Filter: %.2f dB\n', filtered_SNR_NLMS);
fprintf('SNR after RLS Filter: %.2f dB\n', filtered_SNR_RLS);

% Visualization of Results
figure;
subplot(5, 1, 1);
plot(signal);
title('Clean Signal');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 2);
plot(noisy_signal);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-20 20]);
xlim([0 15000]);

subplot(5, 1, 3); 
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
ylim([-1.1 1.1]);
xlim([0 15000]);

% Normalize filtered signals                             (To prevent "Warning: Data clipped when writing file") 
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));

% Save Filtered Signals
audiowrite('Filtered_LMS.wav', output_LMS, fs);
audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
audiowrite('Filtered_RLS.wav', output_RLS, fs);

fprintf('Filtered signals saved as audio files.\n\n');

% Tuning Parementers for tables
fprintf('Sampling rate: %f Hz\n', fs);
fprintf('Filter order (LMS, NLMS, RLS): %f \n', M);
fprintf('Step size (LMS): %f \n', mu_LMS);
fprintf('Step size (NLMS): %f \n', mu_NLMS);
fprintf('Forgetting factor (RLS): %f \n', lambda);


% Crop the figure and save as PDF
tightfig();
saveas(gcf, 'baseline_sim.pdf');


% sound(signal, fs) %play clean audio 
% sound(noise, fs) %play noise 
% sound(noisy_signal, fs) %play clean + noisy audio 

%% Clear workspace and close figures
clear all;
close all;
clc;

% Define the base path for your folders containing the data
dataPath = "C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Data from AI heathway\Data_ANC\Experiment_Data\Hospital Ambient Noises\NHS\1";
folders = dir(fullfile(dataPath, "*"));

% Filter Parameters
M = 12; % Filter order (adjusted to match first code)

% Loop through folders to load data files and process
for folder = folders'
    if folder.isdir
        folderPath = fullfile(dataPath, folder.name);
        noisyFile = fullfile(folderPath, "primary.wav");
        referenceFile = fullfile(folderPath, "secondary.wav");
        cleanFile = fullfile(folderPath, "ZCH0019.wav");
        
        if isfile(noisyFile) && isfile(referenceFile) && isfile(cleanFile)
            [d, Fs] = audioread(noisyFile);
            [x, ~] = audioread(referenceFile);
            [clean, ~] = audioread(cleanFile);
            
            % Calculate initial SNR for noisy signal
            initial_SNR = 10 * log10(sum(clean.^2) / sum((d - clean).^2));
            
            % Pad noisy signal
            N = length(d); % Signal length
            padded_signal = [zeros(M-1, 1); d]; 

            % *************************************************************************
            % LMS Filter
            mu_LMS = 0.1; % Step size for LMS (adjusted from first code)
            w_LMS = zeros(M, 1); % Initialize filter weights
            output_LMS = zeros(N, 1); % Filter output

            for n = 1:N
                u_vect = padded_signal(n:n+M-1); % Current input vector
                e = clean(n) - w_LMS' * u_vect; % Error signal
                w_LMS = w_LMS + mu_LMS * e * u_vect; % Update weights
                output_LMS(n) = w_LMS' * u_vect; % Filtered output
            end

            % *************************************************************************
            % NLMS Filter
            mu_NLMS = 1; % Step size for NLMS (kept from first code)
            w_NLMS = zeros(M, 1); % Initialize filter weights
            output_NLMS = zeros(N, 1); % Filter output
            Eps = 1e-5; % Stability constant (adjusted from first code)

            for n = 1:N
                u_vect = padded_signal(n:n+M-1); % Current input vector
                mu_adapt = mu_NLMS / (Eps + norm(u_vect)^2); % Adaptive step size
                e = clean(n) - w_NLMS' * u_vect; % Error signal
                w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
                output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
            end

            % *************************************************************************
            % RLS Filter
            lambda = 1 - 1 / (0.1 * M); % Forgetting factor  =  0.1666 
            delta = 0.01; % Initialization constant
            P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
            w_RLS = zeros(M, 1); % Initialize filter weights
            padded_signal = [sqrt(delta) * randn(M-1, 1); d]; % Pad noisy signal
            output_RLS = zeros(N, 1); % Filter output

            for n = 1:N
                u_vect = padded_signal(n:n+M-1); % Current input vector
                PI = P * u_vect; % Intermediate calculation
                gain_k = PI / (lambda + u_vect' * PI); % Gain
                e = clean(n) - w_RLS' * u_vect; % Error signal
                w_RLS = w_RLS + gain_k * e; % Update weights
                P = P / lambda - gain_k * (u_vect' * P) / lambda; % Update P matrix
                output_RLS(n) = w_RLS' * u_vect; % Filtered output
            end

            % *************************************************************************
            % Calculate SNR Improvement
            filtered_SNR_LMS = 10 * log10(sum(clean.^2) / sum((clean - output_LMS).^2));
            filtered_SNR_NLMS = 10 * log10(sum(clean.^2) / sum((clean - output_NLMS).^2));
            filtered_SNR_RLS = 10 * log10(sum(clean.^2) / sum((clean - output_RLS).^2));
        end
    end
end

% Display Results
fprintf('Initial SNR (before filtering): %.2f dB\n', initial_SNR);
fprintf('SNR after LMS Filter: %.2f dB\n', mean(filtered_SNR_LMS));
fprintf('SNR after NLMS Filter: %.2f dB\n', mean(filtered_SNR_NLMS));
fprintf('SNR after RLS Filter: %.2f dB\n', mean(filtered_SNR_RLS));

% *************************************************************************
% Visualization of Results
figure;
subplot(5, 1, 1);
plot(clean);
title('Clean Signal');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(5, 1, 2);
plot(d);
title('Noisy Signal');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(5, 1, 3);
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');

% *************************************************************************
% Normalize filtered signals
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));

% Save Filtered Signals
fs = Fs;      % Sampling frequency
audiowrite('Filtered_LMS.wav', output_LMS, fs);
audiowrite('Filtered_NLMS.wav', output_NLMS, fs);
audiowrite('Filtered_RLS.wav', output_RLS, fs);

fprintf('Filtered signals saved as audio files.\n');

% Crop the figure and save as PDF
tightfig();
saveas(gcf, 'filtered_signals_comparison.pdf');

%%
% Simulate Signal and Noise
fs = 8000; % Sampling frequency
t = 0:1/fs:2; % Time vector (2 seconds)
signal = sin(2 * pi * 10 * t)'; % Simulated clean signal (10 Hz sine wave)

% Generate noise at -20 dB SNR
signal_power = mean(signal.^2);
SNR_dB = -20; % Desired SNR in dB
SNR_linear = 10^(SNR_dB/10);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power) * randn(size(t))';

noisy_signal = signal + noise; % Combine signal and noise

% Define SNR improvement tracking
bestSNR = struct('LMS', -Inf, 'NLMS', -Inf, 'RLS', -Inf);
bestParams = struct('LMS', [], 'NLMS', [], 'RLS', []);

% Define parameter search space
mu_values = [0.001, 0.005, 0.01, 0.05, 0.1];  % Step sizes for LMS & NLMS
filterOrders = [5, 10, 20, 50];  % Filter orders
lambda_values = [0.9, 0.95, 0.99, 0.999];  % Forgetting factors for RLS

% Clean signal for SNR calculation
clean = signal;  % The clean signal is the sine wave

% Run through the filters
for M = filterOrders
    for mu = mu_values
        % LMS Filtering
        [y_LMS, ~] = lms_filter(noisy_signal, noisy_signal, M, mu);
        if all(isfinite(y_LMS))  % Check for finite values
            snr_LMS = snr(clean, clean - y_LMS);
            if snr_LMS > bestSNR.LMS
                bestSNR.LMS = snr_LMS;
                bestParams.LMS = [M, mu];
            end
        end

        % NLMS Filtering
        [y_NLMS, ~] = nlms_filter(noisy_signal, noisy_signal, M, mu);
        if all(isfinite(y_NLMS))  % Check for finite values
            snr_NLMS = snr(clean, clean - y_NLMS);
            if snr_NLMS > bestSNR.NLMS
                bestSNR.NLMS = snr_NLMS;
                bestParams.NLMS = [M, mu];
            end
        end
    end

    for lambda = lambda_values
        % RLS Filtering
        [y_RLS, ~] = rls_filter(noisy_signal, noisy_signal, M, lambda);
        if all(isfinite(y_RLS))  % Check for finite values
            snr_RLS = snr(clean, clean - y_RLS);
            if snr_RLS > bestSNR.RLS
                bestSNR.RLS = snr_RLS;
                bestParams.RLS = [M, lambda];
            end
        end
    end
end

% Display the best parameters
fprintf("Best LMS: Filter Order = %d, Step Size = %.4f, SNR = %.2f dB\n", bestParams.LMS(1), bestParams.LMS(2), bestSNR.LMS);
fprintf("Best NLMS: Filter Order = %d, Step Size = %.4f, SNR = %.2f dB\n", bestParams.NLMS(1), bestParams.NLMS(2), bestSNR.NLMS);
fprintf("Best RLS: Filter Order = %d, Forgetting Factor = %.4f, SNR = %.2f dB\n", bestParams.RLS(1), bestParams.RLS(2), bestSNR.RLS);


%%
clc;
clear;
close all;

% Define paths
dataPath = "C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Data from AI heathway\Data_ANC\Experiment_Data\Hospital Ambient Noises\NHS\1";
folders = dir(fullfile(dataPath, "*"));

% Define SNR improvement tracking for NLMS
bestSNR_NLMS = -Inf;
bestParams_NLMS = [];

% Define NLMS parameter search space
mus = 1e-3 * (2.^(0:9));  % Step sizes for NLMS
ps = 1:10;  % Filter orders

for folder = folders'
    if folder.isdir
        folderPath = fullfile(dataPath, folder.name);
        noisyFile = fullfile(folderPath, "primary.wav");
        referenceFile = fullfile(folderPath, "secondary.wav");
        cleanFile = fullfile(folderPath, "ZCH0019.wav");
        
        if isfile(noisyFile) && isfile(referenceFile) && isfile(cleanFile)
            [d, Fs] = audioread(noisyFile);
            [x, ~] = audioread(referenceFile);
            [clean, ~] = audioread(cleanFile);

            % Loop through different configurations for NLMS
            SNRs = zeros(length(mus), length(ps));
            for i = 1:length(mus)
                for j = 1:length(ps)
                    
                    % Parameters
                    mu = mus(i);
                    p = ps(j);
                    
                    % Initialization for NLMS
                    w = zeros(p, 1);
                    x_n = zeros(p, 1);  % Signal fragment to be filtered
                    e = zeros(length(d), 1);

                    % NLMS Filtering
                    for k = 1:length(d)
                        x_n(2:end) = x_n(1:end-1);
                        x_n(1) = x(k);
                        sig = sum(x_n.^2);
                        e(k) = d(k) - w' * x_n;
                        w = w + (mu / (sig + 1e-10)) * x_n * e(k);
                    end

                    % SNR computation
                    snr_NLMS = 10 * log10(sum(clean.^2) / sum((clean - e).^2));
                    SNRs(i, j) = snr_NLMS;
                    
                    % Track best SNR and parameters
                    if snr_NLMS > bestSNR_NLMS
                        bestSNR_NLMS = snr_NLMS;
                        bestParams_NLMS = [p, mu];
                    end
                end
            end

            % Display the best parameters
            fprintf("Best NLMS: Filter Order = %d, Step Size = %.4f, SNR = %.2f dB\n", bestParams_NLMS(1), bestParams_NLMS(2), bestSNR_NLMS);

            % Plot time evolution of the filter weights for the best SNR configuration
            opt_mu = bestParams_NLMS(2);  % Optimal step-size parameter
            opt_p = bestParams_NLMS(1);  % Optimal filter order
            
            % Initialization for optimal configuration
            w = zeros(opt_p, length(d) + 1);
            x_n = zeros(opt_p, 1);  % Signal fragment to be filtered
            e = zeros(length(d), 1);

            % Filtering with optimal configuration
            for k = 1:length(d)
                x_n(2:end) = x_n(1:end-1);
                x_n(1) = x(k);
                sig = sum(x_n.^2);
                e(k) = d(k) - w(:, k)' * x_n;
                w(:, k + 1) = w(:, k) + (opt_mu / (sig + 1e-10)) * x_n * e(k);
            end

            % Plot weights evolution
            figure;
            plot(w');
            grid on;
            title('NLMS Weights Evolution');
            xlabel('Cycle (n)');
            ylabel('Magnitude');
            legend(arrayfun(@(i) sprintf('w_%d(n)', i - 1), 1:opt_p, 'UniformOutput', false));
            
            % Plot signal and filtered output
            figure;
            plot(clean);
            hold on;
            plot(e, 'r--');
            hold off;
            legend('Clean', 'Filtered');
            grid on;
            xlabel('Time (n)');
            ylabel('Amplitude');
        end
    end
end



function tightfig()
    % Tighten the figure by removing excess whitespace
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end