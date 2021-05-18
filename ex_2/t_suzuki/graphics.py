import matplotlib.pyplot as plt

def draw_freq_res(x, amp, phase, N):
    """
    x : array_like
        frequency label
    amp : array_like
        frequecy response db
    phase : array_like
        frequency respose phase
    N : int
        the number of filter coefficients
    """

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    fig, axes = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.6)

    axes[0].plot(x, amp[0:N//2+1])
    axes[1].plot(x, phase[0:N//2+1])
    axes[0].set_title("Filter amplitude")
    axes[1].set_title("Filter phase")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Amplitude [dB]")
    axes[1].set_ylabel("Phase [deg]")

    axes[0].grid(ls="--")
    axes[1].grid(ls="--")

    plt.show(block=True)
    return fig

def draw_spec(spec, f_spec, sr, time):
    """
    spec : spectrogram data
    f_spec : filtered spectrogram data
    sr : sampling rate
    time : time end point
    """

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.subplots_adjust(wspace=0.4)

    im0 = axes[0].imshow(spec, extent=[0, time, 0, (sr/2) / 1000], aspect="auto", cmap="rainbow")
    cbar = fig.colorbar(im0, ax = axes[0])
    im1 = axes[1].imshow(f_spec, extent=[0, time, 0, (sr/2) / 1000], aspect="auto", cmap="rainbow")
    cbar = fig.colorbar(im1, ax = axes[1])

    axes[0].set(
            title="Original Spectrogram",
            xlabel="Time [s]",
            ylabel="Frequency [kHz]",
    )
    axes[1].set(
            title="Filtered Spectrogram",
            xlabel="Time [s]",
            ylabel="Frequency [kHz]",
    )

    plt.show(block=True)
    return fig
