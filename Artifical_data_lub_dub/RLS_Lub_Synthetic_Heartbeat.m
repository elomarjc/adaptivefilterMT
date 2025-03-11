%% Enhanced Synthetic Heartbeat with Clear "Lub-Dub" Sound
clc;
close all;
clear all;

%% Parameters for Heartbeat Sound
fs = 1000; % Sampling frequency (in Hz)
duration = 15; % Duration of the signal (in seconds)
t = 0:1/fs:duration; % Time vector
heartbeat_bpm = 60; % Heartbeat rate (60 beats per minute)
heartbeat_period = 60 / heartbeat_bpm; % Time between beats in seconds

% Define the "lub-dub" structure of a single heartbeat
lub_duration = 0.1; % Duration of "lub" sound (in seconds)
dub_duration = 0.15; % Duration of "dub" sound (in seconds)
pause_duration = heartbeat_period - (lub_duration + dub_duration); % Pause between heartbeats

% Create the "lub-dub" pattern using Gaussian pulses
t_lub = 0:1/fs:lub_duration; % Time for "lub" sound
t_dub = 0:1/fs:dub_duration; % Time for "dub" sound
lub_sound = gauspuls(t_lub, 150, 0.5); % Simulated "lub" sound with Gaussian pulse
dub_sound = gauspuls(t_dub, 80, 0.5);  % Simulated "dub" sound with lower frequency Gaussian pulse

% Create one full heartbeat cycle
heartbeat_cycle = [lub_sound, ...
                   zeros(1, round(pause_duration*fs/2)), ...
                   dub_sound, ...
                   zeros(1, round(pause_duration*fs/2))];

% Calculate how many cycles fit in the signal duration
num_heartbeats = floor(duration / heartbeat_period);

% Replicate the heartbeat cycle to fill the total duration
heartbeat_signal = repmat(heartbeat_cycle, 1, num_heartbeats);

% Ensure the heartbeat signal matches the length of the time vector
heartbeat_signal = heartbeat_signal(1:length(t));

% Normalize the heartbeat signal
heartbeat_signal = normalize(heartbeat_signal, 'range', [-1 1]);

% % Add small ambient noise for realism
% noise_amplitude = 0.05;
% heartbeat_signal_noisy = heartbeat_signal + noise_amplitude * randn(size(t));

% % Plot the enhanced synthetic heartbeat signal
% figure;
% plot(t, heartbeat_signal_noisy);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Enhanced Synthetic Heartbeat with "Lub-Dub" Sound');
% axis tight;

% Save the synthetic heartbeat signal to a .wav file (optional)
audiowrite('clean_synthetic_heartbeat.wav', heartbeat_signal, fs);

%% Simulate Ambient Noise and Corrupted Signal
% Simulate ambient noise and mix it with the heartbeat signal
desired_snr_db = -20; % Desired SNR for noisy heartbeat (< -15 dB as per your requirement)
signal_power = rms(heartbeat_signal)^2;
noise_power = signal_power / (10^(desired_snr_db/10));
ambient_noise = sqrt(noise_power) * randn(size(t)); % Generate ambient noise
noisy_heartbeat_signal = heartbeat_signal + ambient_noise;

% % Plot the noisy heartbeat signal
% figure;
% plot(t, noisy_heartbeat_signal);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Noisy Heartbeat Signal (SNR < -15 dB)');
% axis tight;

% Save the synthetic heartbeat signal to a .wav file (optional)
audiowrite('noisy_synthetic_heartbeat.wav', noisy_heartbeat_signal, fs);

%% RLS ANC with Enhanced Heartbeat

% Use ambient noise as the reference signal (g2) for ANC
g2 = ambient_noise';  % Reference noise input

% Use noisy heartbeat signal (x) as the corrupted input signal
x = noisy_heartbeat_signal';  % Corrupted signal (noisy heartbeat)

% Simulate clean signal (s) for comparison (optional)
s = heartbeat_signal';  % Clean heartbeat signal

% Time vector for plotting
Ts = 1/fs; % Sampling period
k = length(t) - 1; % Max. sample for 15 seconds recording
t = [0:k]*Ts; % Time vector

%% RLS Filter Parameters
M = 7; % Filter length
lambda = 1; % Forgetting factor
lambdainv = 1/lambda;
delta = 5; % Small positive constant
deltainv = 1/delta;

% Filter Initialization
N = length(x); % Number of samples in the input signal x
w = zeros(M,1); % Initialize filter coefficients
P = deltainv*eye(M); % Inverse correlation matrix
e = zeros(N,1); % Error signal

% Ensure g2 and x are column vectors
g2 = g2(:);
x = x(:);
m = 0; % Variable to count the number of iterations performed

%% ANC using RLS Algorithm
for i = M:N
    % Obtain reference input vector length M
    y = g2(i:-1:i-M+1);
    
    % Error signal equation - Estimated output signal
    e(i) = x(i) - w' * y;
    
    % Filter gain vector update
    k = (P * y) / (lambda + y' * P * y);
    
    % Inverse correlation matrix update
    P = (P - k * y' * P) * lambdainv;
    
    % Filter coefficients update
    w = w + k * e(i);
    
    m = m + 1; % Count iterations
    w1(m,:) = w(:); % Store history of filter coefficients
end

% Normalize the estimated signal
e = normalize(e, 'range', [-1 1]);

%% Plot the ANC Results
figure;
subplot(3, 1, 1);
plot(s);
title('Clean Heartbeat Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(s)]);

subplot(3, 1, 2);
plot(x);
title('Noisy Heartbeat Signal');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(x)]);

subplot(3, 1, 3);
plot(e);
title('Recovered Signal using RLS');
xlabel('Sample Number');
ylabel('Amplitude');
xlim([0 length(e)]);

sgtitle('ANC Using RLS Algorithm');

% % Comparison of Clean Signal and ANC Output
% figure;
% plot(s, 'r');
% hold on;
% plot(e, '--g');
% title('Comparison of Clean Heartbeat Signal and ANC Output');
% legend('Clean Heartbeat Signal', 'ANC Output');
% axis tight;
% 
% % Time comparison
% figure;
% plot(t(1:length(s)), s, 'r'); % Clean heartbeat
% hold on;
% plot(t(1:length(e)), e, '--g'); % ANC output
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Comparison of Clean Heartbeat and ANC Output (Time Domain)');
% legend('Clean Heartbeat', 'ANC Output');
% axis tight;
% grid on;
    
% % Adaptation of filter coefficients over time
% figure;
% plot(w1);
% title('Adaptation of Filter Coefficients Over Time');
% xlim([0 600]);
% axis tight;

% To listen to the signals (optional)
% sound(x, fs);  % Play the corrupted signal
% sound(e, fs);  % Play the recovered signal (ANC output)
% sound(s, fs);  % Play the clean signal

%% Save Filtered Signal
audiowrite('filtered_synthetic_heartbeat.wav', e, fs);

fprintf('Filtered RLS signal saved as audio file.\n');

% Save visualization as PDF
tightfig();
saveas(gcf, 'filtered_synthetic_heartbeat.pdf');

%% Function for Tightening Figure Layout
function tightfig()
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
end