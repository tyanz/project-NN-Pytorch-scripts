import os
import librosa
import pysptk
import soundfile as sf
import wave
import pickle
import copy
import numpy as np
from scipy import signal, io
import argparse
from tqdm import tqdm

nchannels=1
sampwidth=2
nframes=0
comptype='NONE'
compname='NONE'

class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 virtual_sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=None,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 pitch_floor=None,
                 pitch_ceiling=None,
                 clip_norm=True,
                 use_admm=False,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_top_db=40,
                 do_normalize_volume=False,
                 do_clip_wav=False,
                 verbose=True,
                 **kwargs):

        print("> Setting up Audio Processor...")

        self.sample_rate = sample_rate
        self.virtual_sample_rate = virtual_sample_rate if  virtual_sample_rate else sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db

        """

        hop_length = int(frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(frame_length_ms / 1000.0 * self.sample_rate)

        Example:

        hop_length | frame_shift_ms | win_length | frame_length_ms
        275        |  12.50 ms      | 1100       | 50 ms
        256        |  11.61 ms      | 1024       | 46.44 ms
        """

        self.hop_length = hop_length
        self.win_length = win_length

        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.use_admm = use_admm
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = 0 if mel_fmin is None else mel_fmin
        self.mel_fmax = mel_fmax
        self.pitch_floor = 50 if pitch_floor is None else pitch_floor
        self.pitch_ceiling = 500 if pitch_ceiling is None else pitch_ceiling
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_top_db = trim_top_db
        self.n_fft = (self.num_freq - 1) * 2
        self.do_normalize_volume = do_normalize_volume
        self.do_clip_wav = do_clip_wav

        if verbose:
            members = vars(self)
            for key, value in members.items():
                print(" | > {}:{}".format(key, value))
        

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear_old(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _mel_to_linear(self, mel_spec):
        m = self._build_mel_basis()
        m_t = np.transpose(m)
        p = np.matmul(m, m_t)
        d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]
        inv_mel = np.matmul(m_t, np.diag(d))
        return np.matmul(inv_mel, mel_spec)

    def _build_mel_basis(self, ):
        n_fft = (self.num_freq - 1) * 2
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.virtual_sample_rate // 2
        return librosa.filters.mel(
            self.virtual_sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        S *= 20
        S -= self.ref_level_db
        S_norm = ((S - self.min_level_db) / - self.min_level_db)
        if self.symmetric_norm:
            S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
            if self.clip_norm :
                S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
        else:
            S_norm = self.max_norm * S_norm
            if self.clip_norm:
                S_norm = np.clip(S_norm, 0, self.max_norm)
        return S_norm

    def _denormalize(self, S):
        """denormalize values"""

        S_denorm = S
        if self.symmetric_norm:
            if self.clip_norm:
                S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm) 
            S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
        else:
            if self.clip_norm:
                S_denorm = np.clip(S_denorm, 0, self.max_norm)
            S_denorm = (S_denorm * -self.min_level_db /
                self.max_norm) + self.min_level_db
        S_denorm += self.ref_level_db
        S_denorm *= 0.05
        return S_denorm

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D))
        S = self._normalize(S) if self.signal_norm else S
        return S

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))  # shape: [#fft+1, #frames], eg: [1025, 312]
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))  # shape:[#mel, #frames], eg: [80, 312]
        S = self._normalize(S) if self.signal_norm else S
        return S

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram) if self.signal_norm else spectrogram
        S = self._db_to_amp(S)  # Convert back to linear
        # Reconstruct phase
        if self.use_admm:
            W = self._admm_griffin_lim(S**self.power)
        else:
            W = self._griffin_lim(S**self.power)
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(W)
        else:
            return W

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram) if self.signal_norm else mel_spectrogram
        S = self._db_to_amp(D)
        S = self._mel_to_linear(S)  # Convert back to linear
        # Reconstruct phase
        if self.use_admm:
            W = self._admm_griffin_lim(S**self.power)
        else:
            W = self._griffin_lim(S**self.power)
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(W)
        else:
            return W

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _admm_griffin_lim(self, S, rho=0.1):
        z = S*np.exp(2 * np.pi * 1j * np.random.rand(S.shape[0], S.shape[1]))
        u = np.zeros(S.shape)
        for i in range(self.griffin_lim_iters):
            x = S * np.exp(1j * np.angle(z-u))
            v = x + u
            z = (rho * v + self._stft(self._istft(v)))/(1+rho)
            u = u + x - z
        return self._istft(x)

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.virtual_sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.1 sec margin """
        margin = int(self.virtual_sample_rate * 0.1)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=self.trim_top_db, frame_length=self.win_length, hop_length=self.hop_length)[0]

    def load_wav(self, filename, resample=False):
        wav, source_sr = sf.read(filename)
        # re-sample the wav if needed
        if source_sr is not None and self.sample_rate != source_sr:
            if resample is True:
                wav = librosa.resample(wav, source_sr, self.sample_rate)
            else:
                raise ValueError('Source sample rate is not equal to config sample rate!')

        # Apply the preprocessing: normalize volume and shorten long silences
        wav = self.clip_wav(wav) if self.do_clip_wav else wav
        wav = self.normalize_volume(wav, increase_only=True) if self.do_normalize_volume else wav
        wav = self.trim_silence(wav) if self.do_trim_silence else wav
        # wav = self.trim_long_silences(wav)

        return wav

    def pcm2wav(self, pcm_file, wav_dir):
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir, exist_ok=True)
        bn = os.path.basename(pcm_file)
        with open(pcm_file, 'rb') as pcmfile:
            pcmdata = pcmfile.read()
        wav_file_path = os.path.join(wav_dir, bn.split('.')[0] + ".wav")
        wavfile = wave.open(wav_file_path, 'wb')
        wavfile.setparams((nchannels, sampwidth, self.sample_rate, nframes, comptype, compname))
        wavfile.writeframes(pcmdata)
        wavfile.close()
        return wav_file_path

    def wav2pcm(self, wav_file, pcm_dir):
        if not os.path.exists(pcm_dir):
            os.makedirs(pcm_dir, exist_ok=True)

        bn = os.path.basename(wav_file)

        pcm_file = os.path.join(pcm_dir, bn.split('.')[0] + ".pcm")

        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)
        pcm = np.asarray(wav*(2**15), dtype=np.int16)

        pcm.tofile(pcm_file)
        return pcm_file

    def wav2mel(self, wav_file, mel_dir):
        if not os.path.exists(mel_dir):
            os.makedirs(mel_dir, exist_ok=True)
        bn = os.path.basename(wav_file)
        mel_file_path = os.path.join(mel_dir, bn.split('.')[0] + ".mspec")
        wav = self.load_wav(wav_file)
        mel = self.melspectrogram(wav).T.astype(np.float32)  # (T, n_mels)
        mel.tofile(mel_file_path)
        return mel_file_path

    def wav2linear(self, wav_file, linear_dir):
        if not os.path.exists(linear_dir):
            os.makedirs(linear_dir, exist_ok=True)
        bn = os.path.basename(wav_file)
        linear_file_path = os.path.join(linear_dir, bn.split('.')[0] + ".linear.npy")
        wav = self.load_wav(wav_file)
        linear = self.spectrogram(wav).T.astype(np.float32)  # (T, n_linear)
        np.save(linear_file_path, linear)
        return linear_file_path
    

    def get_duration(self, audio_file, sr=22050):
        return librosa.get_duration(filename=audio_file, sr=sr)


    @staticmethod
    def normalize_volume(wav, target_dBFS=-30, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
        if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
            return wav
        return wav * (10 ** (dBFS_change / 20))

    @staticmethod
    def clip_wav(wav, q_min=0.005, q_max=0.995):
        p_min = np.quantile(wav, q_min, interpolation="nearest")
        p_max = np.quantile(wav, q_max, interpolation="nearest")
        iqr = p_max - p_min
        return np.clip(wav, p_min - iqr, p_max + iqr)

    def wav2f0(self, wav_file, f0_dir):
        sr, x = io.wavfile.read(wav_file)
        bn = os.path.basename(wav_file)
        f0_file_path = os.path.join(f0_dir, bn.split('.')[0] + ".f0")
        f0 =  pysptk.swipe(
                x.astype(np.float64), 
                fs=sr, 
                hopsize=self.hop_length,
                min=self.pitch_floor,
                max=self.pitch_ceiling,
                otype="f0").astype(np.float32)
        f0.tofile(f0_file_path)
        return f0_file_path

def main(args):
    wav_dir = args.wav_dir
    mel_dir = args.mel_dir
    f0_dir = args.f0_dir
    ap = AudioProcessor(
                 sample_rate=16000,
                 num_mels=80,
                 min_level_db=-100,
                 hop_length=80,
                 win_length=320,
                 ref_level_db=20,
                 num_freq=161,
                 preemphasis=0.98,
                 signal_norm=True,
                 symmetric_norm=False,
                 max_norm=1,
                 mel_fmin=75,
                 mel_fmax=8000,
                 pitch_floor=75,
                 pitch_ceiling=500,
                 clip_norm=True,
                 do_trim_silence=False,
                 verbose=True)

    for fn in tqdm(os.listdir(wav_dir)):
        if not fn.endswith(".wav"):
            continue
        wav_file = os.path.join(wav_dir, fn)
        
        mel_file = os.path.join(mel_dir, fn.split('.')[0]+'.mspec')
        if not os.path.exists(mel_file):
            mel_file = ap.wav2mel(wav_file, mel_dir)
        
        f0_file = os.path.join(f0_dir, fn.split('.')[0]+'.f0')
        if not os.path.exists(f0_file):
            f0_file = ap.wav2f0(wav_file, f0_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--wav_dir",
            type=str,
            help="input directroy storing waveforms",
            default='')
    parser.add_argument(
            "--mel_dir",
            type=str,
            help="output directroy storing mel-spectrograms",
            default='')
    parser.add_argument(
            "--f0_dir",
            type=str,
            help="output directroy storing f0s",
            default='')
    args = parser.parse_args()
    main(args)
