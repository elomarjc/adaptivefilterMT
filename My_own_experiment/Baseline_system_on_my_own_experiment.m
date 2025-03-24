%% Clear workspace and close figures
clear all;
close all;
clc;

% Define the base path for your folders containing the data
dataPath = "C:\Users\eloma\Desktop\Universitet\OneDrive - Aalborg Universitet\Universitet\9. Semester - ES9\Long Thesis\Matlab\adaptivefilterMT\My_own_experiment\NHS data\speech";
folders = dir(fullfile(dataPath, "*"));

% Filter Parameters
M = 12; % Filter order (adjusted to match first code)

% Loop through folders to load data files and process
for folder = folders'
    if folder.isdir
        folderPath = fullfile(dataPath, folder.name);
        noisyFile = fullfile(folderPath, "Primary.wav");
        referenceFile = fullfile(folderPath, "Secondary.wav");

        if isfile(noisyFile) && isfile(referenceFile)
            [d, Fs] = audioread(noisyFile);  % Noisy signal with heartbeat    % originally d 
            [x, ~] = audioread(referenceFile); % Reference noise signal       % originally x
            
            % Calculate initial SNR for noisy signal (using reference noise)
            initial_SNR = 10 * log10(sum(x.^2) / sum((d - x).^2));
            
            % Pad noisy signal
            N = length(d); % Signal length
            padded_signal = [zeros(M-1, 1); d]; 

            % *************************************************************************
            % LMS Filter
            mu_LMS = 0.9; % Step size for LMS (adjusted from first code)
            w_LMS = zeros(M, 1); % Initialize filter weights
            output_LMS = zeros(N, 1); % Filter output

            for n = 1:N
                u_vect = padded_signal(n:n+M-1); % Current input vector
                % Calculate error signal (based on the noisy signal and filter output)
                e = d(n) - w_LMS' * u_vect; % Error signal using the noisy signal as the desired response
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
                % Calculate error signal (based on the noisy signal and filter output)
                e = d(n) - w_NLMS' * u_vect; % Error signal using the noisy signal as the desired response
                w_NLMS = w_NLMS + mu_adapt * e * u_vect; % Update weights
                output_NLMS(n) = w_NLMS' * u_vect; % Filtered output
            end

            % *************************************************************************
            % RLS Filter
            lambda = 1 - 1 / (0.1 * M); % Forgetting factor  =  0.1666    =  1 - 1 / (0.1 * M)
            delta = 0.01; % Initialization constant
            P = (1 / delta) * eye(M); % Initialize inverse correlation matrix
            w_RLS = zeros(M, 1); % Initialize filter weights
            padded_signal = [sqrt(delta) * randn(M-1, 1); d]; % Pad noisy signal
            output_RLS = zeros(N, 1); % Filter output

            for n = 1:N
                u_vect = padded_signal(n:n+M-1); % Current input vector
                PI = P * u_vect; % Intermediate calculation
                gain_k = PI / (lambda + u_vect' * PI); % Gain
                % Calculate error signal (based on the noisy signal and filter output)
                e = d(n) - w_RLS' * u_vect; % Error signal using the noisy signal as the desired response
                w_RLS = w_RLS + gain_k * e; % Update weights
                P = P / lambda - gain_k * (u_vect' * P) / lambda; % Update P matrix
                output_RLS(n) = w_RLS' * u_vect; % Filtered output
            end

            % *************************************************************************
            % Calculate SNR Improvement (using the noisy signal as the clean signal)
            filtered_SNR_LMS = 10 * log10(sum(d.^2) / sum((d - output_LMS).^2));
            filtered_SNR_NLMS = 10 * log10(sum(d.^2) / sum((d - output_NLMS).^2));
            filtered_SNR_RLS = 10 * log10(sum(d.^2) / sum((d - output_RLS).^2));
        end
    end
end

% Normalize signals
output_LMS = output_LMS / max(abs(output_LMS));
output_NLMS = output_NLMS / max(abs(output_NLMS));
output_RLS = output_RLS / max(abs(output_RLS));
d = d / max(abs(d));  
x = x / max(abs(x));  

% Display Results
fprintf('Initial SNR (before filtering): %.2f dB\n', initial_SNR);
fprintf('SNR after LMS Filter: %.2f dB\n', mean(filtered_SNR_LMS));
fprintf('SNR after NLMS Filter: %.2f dB\n', mean(filtered_SNR_NLMS));
fprintf('SNR after RLS Filter: %.2f dB\n', mean(filtered_SNR_RLS));

% *************************************************************************
% Visualization of Results
figure;
subplot(5, 1, 1);
plot(d);
title('Heart + Noise signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

subplot(5, 1, 2);
plot(x);
title('Reference Noise Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

subplot(5, 1, 3);
plot(output_LMS);
title('Filtered Signal (LMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

subplot(5, 1, 4);
plot(output_NLMS);
title('Filtered Signal (NLMS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

subplot(5, 1, 5);
plot(output_RLS);
title('Filtered Signal (RLS)');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

% *************************************************************************

% Save Filtered Signals
fs = Fs;      % Sampling frequency
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
saveas(gcf, 'MOE_heartbeat_speech.pdf');

function tightfig()
    % Tighten the figure by removing excess whitespace
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end
