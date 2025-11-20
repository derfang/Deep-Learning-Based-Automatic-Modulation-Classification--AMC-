% ---------------------------------------------------------
%  Synthetic Dataset Generator for AMC (BPSK, QPSK, 16QAM)
%  Format: [Batch_Size, Time_Steps, Channels] -> (N, 128, 2)
% ---------------------------------------------------------
clear; clc; close all;

% --- CONFIGURATION ---
mods = {'BPSK', 'QPSK', '16QAM'};
snr_range = -10:2:18;       % SNR from -10dB to 18dB
samples_per_frame = 128;    % 128 samples per "image"
frames_per_mod_snr = 2000;   % Examples per SNR point

% Calculate total number of frames
total_frames = length(mods) * length(snr_range) * frames_per_mod_snr;

% Pre-allocate arrays
% CHANGED: Now [Total_Frames, Samples, 2] for Keras compatibility
% Channel 1 = In-Phase (Real), Channel 2 = Quadrature (Imag)
X = zeros(total_frames, samples_per_frame, 2); 
Y = zeros(total_frames, length(mods)); % One-Hot Labels
Z = zeros(total_frames, 1);            % SNR Labels

counter = 1;

disp(['Starting generation of ', num2str(total_frames), ' frames...']);

% --- GENERATION LOOP ---
for m = 1:length(mods)
    current_mod = mods{m};
    
    for s = 1:length(snr_range)
        current_snr = snr_range(s);
        
        for f = 1:frames_per_mod_snr
            
            % 1. Generate Random Integer Symbols
            if strcmp(current_mod, 'BPSK')
                M = 2;
                data = randi([0 M-1], samples_per_frame, 1);
                tx_sig = pskmod(data, M);
                
            elseif strcmp(current_mod, 'QPSK')
                M = 4;
                data = randi([0 M-1], samples_per_frame, 1);
                tx_sig = pskmod(data, M, pi/4); 
                
            elseif strcmp(current_mod, '16QAM')
                M = 16;
                data = randi([0 M-1], samples_per_frame, 1);
                tx_sig = qammod(data, M, 'UnitAveragePower', true);
            end
            
            % 2. Add Channel Impairments
            
            % A) Random Phase Rotation
            phase_offset = 2*pi*rand; 
            tx_sig = tx_sig .* exp(1j * phase_offset);
            
            % B) Add AWGN (Noise)
            rx_sig = awgn(tx_sig, current_snr, 'measured');
            
            % 3. Store Data (CHANGED FORMAT)
            % We now store time steps in dim 2, and I/Q in dim 3
            X(counter, :, 1) = real(rx_sig); % Channel 1: I
            X(counter, :, 2) = imag(rx_sig); % Channel 2: Q
            
            % 4. Store Label (One-Hot Encoding)
            label_vec = zeros(1, length(mods));
            label_vec(m) = 1;
            Y(counter, :) = label_vec;
            
            % 5. Store SNR
            Z(counter) = current_snr;
            
            counter = counter + 1;
        end
    end
    disp(['Finished generating ', current_mod]);
end

% --- SAVE TO FILE ---
filename = 'my_synthetic_dataset.mat';
save(filename, 'X', 'Y', 'Z', 'mods');

disp(['Success! Data saved to ', filename]);