import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from matplotlib import patches

def zplane(b, a, filename=None):
    z = np.roots(b)
    p = np.roots(a)
    ax = plt.gca()
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(unit_circle)
    plt.plot(np.real(z), np.imag(z), 'bo', fillstyle='none')
    plt.plot(np.real(p), np.imag(p), 'rx')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axis('equal')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')

def freqz_plot(b, a, title):
    w, h = signal.freqz(b, a)
    norm_freq = w / np.pi
    plt.plot(norm_freq, 20 * np.log10(abs(h)))
    plt.title(title)
    plt.xlabel('Normalized Frequency (\times\pi rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)

def impz_plot(b, a, title, num_points=50):
    impulse = np.zeros(num_points)
    impulse[0] = 1
    response = signal.lfilter(b, a, impulse)
    plt.stem(np.arange(len(response)), response)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)

def stepz_plot(b, a, title, num_points=50):
    step = np.ones(num_points)
    response = signal.lfilter(b, a, step)

    plt.stem(np.arange(len(response)), response)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)

s = 0
a = [1]
b1 = [0.4, 0, 0, 0.3]
b2 = [0.8, 0, 0, 0.7]

plt.figure(1)
freqz_plot(b1, a, "Frequency Response of Echo Filter (c = 0.4)")
plt.figure(2)
freqz_plot(b2, a, "Frequency Response of Echo Filter (c = 0.8)")
plt.figure(3)
zplane(b1, a)
plt.title("Zero - Pole Diagram of Echo Filter (c = 0.4)")
plt.grid(True)
plt.figure(4)
zplane(b2, a)
plt.title("Zero - Pole Diagram of Echo Filter (c = 0.8)")
plt.grid(True)
plt.figure(5)
impz_plot(b1, a, "Impulse Response of Echo Filter (c = 0.4)")
plt.figure(6)
impz_plot(b2, a, "Impulse Response of Echo Filter (c = 0.8)")
r = [0.166375, 0, 0.408375, 0, 0.334125, 0, 0.091125]
plt.figure(7)
freqz_plot(r, a, "Frequency Response of Reverb Filter")
plt.figure(8)
zplane(r, a)
plt.title("Zero - Pole Diagram of Reverb Filter")
plt.grid(True)
plt.figure(9)
impz_plot(r, a, "Impulse Response of Reverb Filter")
impulse_gen = np.zeros(100)
impulse_gen[0] = 1
h1 = signal.lfilter(r, a, impulse_gen)
am = [0.166375, 0, 0.408375, 0, 0.334125, 0, 0.091125]
bm = [1]
h2 = signal.lfilter(bm, am, impulse_gen)
x = np.arange(-1, 14)
u = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plt.figure(10)
plt.plot(x, u, 'b.')
temp1 = np.convolve(u, h1[:len(r)])
y1 = temp1[0:15]
plt.plot(x, y1, 'gd')
temp2 = np.convolve(y1, h2[:50])
y2 = temp2[0:15]
plt.plot(x, y2, 'ro', fillstyle='none')
plt.grid(True)
plt.title("Original, Reverbed and Dereverbed Signals")
plt.legend(["u[n] - u[n-5]", "reverbed signal", "dereverbed signal"])
plt.xlabel("n (samples)")
plt.ylabel("Amplitude")
p = np.array([0.5 + 0.8j, 0.5 - 0.8j])
z = np.array([0.8, -0.8])
plt.figure(11)
bz1, az1 = signal.zpk2tf(z, p, 1)
zplane(bz1, az1)
plt.title("Zero - Pole Diagram of Bandpass Filter 1")
plt.grid(True)
plt.figure(12)
freqz_plot(bz1, az1, "Frequency Response of Bandpass Filter 1")
plt.figure(13)
impz_plot(bz1, az1, "Impulse Response of Bandpass Filter 1")
plt.figure(14)
stepz_plot(bz1, az1, "Step Response of Bandpass Filter 1")
configs = [
    {"p": [0.527 + 0.844j, 0.527 - 0.844j], "id": 2, "fig_start": 15},
    {"p": [0.53 + 0.848j, 0.53 - 0.848j], "id": 3, "fig_start": 18},
    {"p": [0.55 + 0.88j, 0.55 - 0.88j], "id": 4, "fig_start": 21},
    {"p": [0.8 + 0.5j, 0.8 - 0.5j], "id": 5, "fig_start": 24}
]

for conf in configs:
    p_curr = np.array(conf["p"])
    bz, az = signal.zpk2tf(z, p_curr, 1)
    plt.figure(conf["fig_start"])
    zplane(bz, az)
    plt.title(f"Zero-Pole Diagram of Bandpass Filter {conf['id']}")
    plt.grid(True)
    plt.figure(conf["fig_start"] + 1)
    freqz_plot(bz, az, f"Frequency Response of Bandpass Filter {conf['id']}")

    if conf["id"] != 5:
        plt.figure(conf["fig_start"] + 2)
        impz_plot(bz, az, f"Impulse Response of Bandpass Filter {conf['id']}")

try:
    fs, Y = wavfile.read('flute_sequence.wav')
    if Y.dtype != np.float32 and Y.dtype != np.float64:
        Y = Y.astype(float) / np.iinfo(Y.dtype).max
    plt.figure(1 + s)
    dt = 1 / 16000
    t = np.arange(0, 9 + dt / 10, dt)
    limit = min(396901, len(Y))
    plt.plot(t[:limit], Y[:limit])
    plt.grid(True)
    plt.title('Flute sequence')
    plt.xlabel('Time (sec)')
    m = max(np.max(Y), abs(np.min(Y)))
    y = Y / m
    n = np.arange(0, 400)
    w = 0.54 + 0.46 * np.cos(2 * np.pi * n / 1000)
    plt.figure(2 + s)
    plt.plot(t[:limit], 50 * y[:limit])
    y2 = y * y
    E = np.convolve(y2, w)
    plt.plot(t[:limit], E[:limit])
    plt.grid(True)
    plt.title('Energy of signal viola series')
    plt.xlabel('Time (sec)')
    dft1 = np.fft.fft(Y)
    plt.figure(3 + s)
    plt.plot(np.abs(dft1[:6000]))
    plt.grid(True)
    plt.title('DFT of Flute sequence')
    plt.xlabel('Frequency (Hz)')
    dnpart = Y[231999:233000]
    plt.figure(4 + s)
    t_part = np.arange(0, (10000 / 441) + (dt * 1000) / 10, dt * 1000)
    plot_len = min(1001, len(dnpart))
    plt.plot(t_part[:plot_len], dnpart[:plot_len])
    plt.grid(True)
    plt.title('Flute D3 note instance')
    plt.xlabel('Time (msec)')
    Dnote = Y[220499:286650]
    dft2 = np.fft.fft(Dnote)
    plt.figure(5 + s)
    plt.plot(np.abs(dft2[:6001]))
    plt.grid(True)
    plt.title('DFT of Flute D3 note')
    plt.xlabel('Frequency (Hz)')
    fs_enote, Enote = wavfile.read('string_note.wav')
    if Enote.dtype != np.float32:
        Enote = Enote.astype(float) / np.iinfo(Enote.dtype).max

    plt.figure(6 + s)
    t_str = np.arange(0, (200 / 147) + dt / 10, dt)
    limit_str = min(60001, len(Enote))
    plt.plot(t_str[:limit_str], Enote[:limit_str])
    plt.grid(True)
    plt.title('string note')
    plt.xlabel('Time (sec)')
    dft3 = np.fft.fft(Enote)
    plt.figure(7 + s)
    plt.plot(np.abs(dft3[:6001]))
    plt.grid(True)
    plt.title('DFT of string E3 note')
    plt.xlabel('Frequency (Hz)')
    pp2 = np.array([0.9957 + 0.0923j, 0.9957 + 0.0923j, 0.9957 - 0.0923j, 0.9957 - 0.0923j])
    zz2 = np.array([0.995 + 0.0923j, 0.9957 + 0.092j, 0.995 - 0.0923j, 0.9957 - 0.092j])
    bzz2, azz2 = signal.zpk2tf(zz2, pp2, 1)
    impulse_len = 1000
    imp = np.zeros(impulse_len);
    imp[0] = 1
    hzon2 = signal.lfilter(bzz2, azz2, imp)
    out2 = signal.convolve(Enote, hzon2, mode='full')
    plt.figure(8 + s)
    t_harm = np.arange(0, (5 / 441) * 1000, dt * 1000)
    segment = out2[69999:70500] if len(out2) > 70500 else out2
    min_len = min(len(t_harm), len(segment))
    plt.plot(t_harm[:min_len], segment[:min_len])
    plt.grid(True)
    plt.title('3rd Harmonic of string E3 note')
    plt.xlabel('Time (msec)')
    dft4 = np.fft.fft(out2)
    plt.figure(9 + s)
    plt.plot(np.abs(dft4[:6001]))
    plt.grid(True)
    plt.title('DFT of filtered string E3 note (3rd harmonic)')
    plt.xlabel('Frequency (Hz)')
    pp4 = np.array([0.9828 + 0.1844j, 0.9828 + 0.1844j, 0.9828 - 0.1844j, 0.9828 - 0.1844j])
    zz4 = np.array([0.982 + 0.1844j, 0.9828 + 0.184j, 0.982 - 0.1844j, 0.9828 - 0.184j])
    bzz4, azz4 = signal.zpk2tf(zz4, pp4, 1)
    hzon4 = signal.lfilter(bzz4, azz4, imp)
    out4 = signal.convolve(Enote, hzon4, mode='full')
    plt.figure(10 + s)
    segment4 = out4[29999:30500] if len(out4) > 30500 else out4
    min_len4 = min(len(t_harm), len(segment4))
    plt.plot(t_harm[:min_len4], segment4[:min_len4])
    plt.grid(True)
    plt.title('5th Harmonic of string E3 note')
    plt.xlabel('Time (msec)')
    dft5 = np.fft.fft(out4)
    plt.figure(11 + s)
    plt.plot(np.abs(dft5[:6001]))
    plt.grid(True)
    plt.title('DFT of filtered string E3 note (5th harmonic)')
    plt.xlabel('Frequency (Hz)')

except FileNotFoundError:
    pass

try:
    fs, piano = wavfile.read('piano_note.wav')
    if piano.dtype != np.float32 and piano.dtype != np.float64:
        piano = piano.astype(float) / np.iinfo(piano.dtype).max

    dt = 1 / fs
    t_piano = np.arange(0, 50000) / fs
    plt.figure(12 + s)
    plt.plot(t_piano, piano[:50000])
    plt.grid(True)
    plt.title('Signal piano note')
    plt.xlabel('Time (sec)')
    as_coeff = [1]
    bech = np.zeros(6616)
    bech1 = 0.85
    bech[6615] = 0.4
    ecim = signal.lfilter(bech, as_coeff, np.r_[1, np.zeros(10000)])
    brev = np.zeros(19846)
    brev[0] = 0.216
    brev[6615] = 0.432
    brev[13230] = 0.288
    brev[19845] = 0.064
    reim = signal.lfilter(brev, as_coeff, np.r_[1, np.zeros(20000)])
    echoed = signal.convolve(piano, ecim, mode='full')
    reverbed = signal.convolve(piano, reim, mode='full')
    plt.figure(13 + s)
    t_echo = np.arange(0, 56615) / fs
    plt.plot(t_echo, echoed[:56615])
    plt.grid(True)
    plt.title('Echoed piano note')
    plt.xlabel('Time (sec)')
    plt.figure(14 + s)
    t_rev = np.arange(0, 69845) / fs
    plt.plot(t_rev, reverbed[:69845])
    plt.grid(True)
    plt.title('Reverbed piano note')
    plt.xlabel('Time (sec)')
    plt.figure(15 + s)
    plt.plot(t_piano, piano[:50000], label='Original')
    plt.plot(t_piano, echoed[:50000], label='Echoed')
    plt.plot(t_piano, reverbed[:50000], label='Reverbed')
    plt.title('Original/Echoed/Reverbed piano note')
    plt.xlabel('Time (sec)')
    plt.legend()
    orft = np.fft.fft(piano)
    ecft = np.fft.fft(echoed)
    reft = np.fft.fft(reverbed)
    f = np.arange(0, 44100)

    def mag2db(x):
        return 20 * np.log10(x + 1e-10)

    plt.figure(16 + s)
    plt.plot(f, mag2db(np.abs(orft[:44100])))
    plt.grid(True)
    plt.title('DFT of signal piano note')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.figure(17 + s)
    plt.plot(f, mag2db(np.abs(ecft[:44100])))
    plt.grid(True)
    plt.title('DFT of echoed piano note')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.figure(18 + s)
    plt.plot(f, mag2db(np.abs(reft[:44100])))
    plt.grid(True)
    plt.title('DFT of reverbed piano note')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.figure(19 + s)
    plt.plot(f, mag2db(np.abs(orft[:44100])), label='Original')
    plt.plot(f, mag2db(np.abs(ecft[:44100])), label='Echoed')
    plt.plot(f, mag2db(np.abs(reft[:44100])), label='Reverbed')
    plt.title('DFT of original/echoed/reverbed piano note')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    wavfile.write('echoed_py.wav', fs, echoed.astype(np.float32))
    wavfile.write('reverbed_py.wav', fs, reverbed.astype(np.float32))
    adrev = np.zeros(19846)
    adrev[0] = 0.216
    adrev[6615] = 0.432
    adrev[13230] = 0.288
    adrev[19845] = 0.064
    bdrev = [1]
    dereverbed = signal.lfilter(bdrev, adrev, reverbed)
    plt.figure(20 + s)
    plt.plot(piano, label='Original')
    plt.plot(dereverbed, label='Dereverbed')
    plt.title('Original and Dereverbed signals')
    plt.xlabel('Time (sec)')
    plt.legend()
    hs = signal.lfilter(bdrev, adrev, reim)
    adrev5 = np.zeros(19861)
    adrev5[0] = 0.216
    adrev5[6620] = 0.432
    adrev5[13240] = 0.288
    adrev5[19860] = 0.064
    dereverbed5 = signal.lfilter(bdrev, adrev5, reverbed)
    plt.figure(21 + s)
    t_short = np.arange(0, 1.5 + 1 / fs, 1 / fs)
    limit5 = min(66151, len(dereverbed5))
    plt.plot(t_short[:limit5], dereverbed5[:limit5])
    plt.grid(True)
    plt.title('Dereverbed signal (5 samples off)')
    plt.xlabel('Time (sec)')
    h5 = signal.lfilter(bdrev, adrev5, reim)
    adrev10 = np.zeros(19876)
    adrev10[0] = 0.216
    adrev10[6625] = 0.432
    adrev10[13250] = 0.288
    adrev10[19875] = 0.064
    dereverbed10 = signal.lfilter(bdrev, adrev10, reverbed)
    plt.figure(22 + s)
    plt.plot(t_short[:limit5], dereverbed10[:limit5])
    plt.grid(True)
    plt.title('Dereverbed signal (10 samples off)')
    plt.xlabel('Time (sec)')
    h10 = signal.lfilter(bdrev, adrev10, reim)
    adrev50 = np.zeros(19996)
    adrev50[0] = 0.216
    adrev50[6665] = 0.432
    adrev50[13330] = 0.288
    adrev50[19995] = 0.064
    dereverbed50 = signal.lfilter(bdrev, adrev50, reverbed)
    plt.figure(23 + s)
    plt.plot(t_short[:limit5], dereverbed50[:limit5])
    plt.grid(True)
    plt.title('Dereverbed signal (50 samples off)')
    plt.xlabel('Time (sec)')
    h50 = signal.lfilter(bdrev, adrev50, reim)
    t_imp = np.arange(0, 9 / 20, dt)
    limit_h = min(len(t_imp), 19846)
    plt.figure(24 + s)
    plt.plot(t_imp[:limit_h], hs[:limit_h])
    plt.grid(True)
    plt.title('Total Impulse Response for Reverb/Dereverb System')
    plt.xlabel('Time (sec)')
    plt.figure(25 + s)
    plt.plot(t_imp[:limit_h], h5[:limit_h])
    plt.grid(True)
    plt.title('Total Impulse Response for Reverb/Dereverb (5 samples off) System')
    plt.figure(26 + s)
    plt.plot(t_imp[:limit_h], h10[:limit_h])
    plt.grid(True)
    plt.title('Total Impulse Response for Reverb/Dereverb (10 samples off) System')
    plt.figure(27 + s)
    plt.plot(t_imp[:limit_h], h50[:limit_h])
    plt.grid(True)
    plt.title('Total Impulse Response for Reverb/Dereverb (50 samples off) System')

except FileNotFoundError:
    pass

plt.show()