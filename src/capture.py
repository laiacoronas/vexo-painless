"""Record a Vexo Open 79 session to Ogg + JSON, optionally with Deepgram text.

Run from the repository root:
    python -m src.capture --seconds 15
    python -m src.capture --seconds 15 --text --language es
    python -m src.capture --sim --seconds 15

Requires a sibling folder named vexo-open-kit, cloned from the official kit.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "vexo-open-kit"
BAND_NUMBER = 79  # Our hackathon team's device; never scan for other bands.


def main():
    parser = argparse.ArgumentParser(description="Capture an audio session from Vexo Open 79")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--language", default="es", help="Deepgram language, e.g. es or en")
    parser.add_argument("--text", action="store_true", help="Send audio to Deepgram to get transcription")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sim", action="store_true", help="Use the kit's public-domain sample audio")
    group.add_argument("--usb", action="store_true", help="Use USB instead of Bluetooth")
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 120:
        parser.error("--seconds must be between 0 and 120")

    if not (KIT / "examples" / "python" / "speech.py").exists():
        parser.error("Missing vexo-open-kit/. Clone the official kit into this project root first.")

    # The kit's own speech.py and ogg_opus.py import its BLE library via kit.py.
    sys.path.insert(0, str(KIT / "examples" / "python"))
    import speech
    from ogg_opus import OggOpusWriter
    from vexo_open.sync import Band

    transport = "sim" if args.sim else ("serial" if args.usb else "ble")
    connect_options = {}
    seconds = args.seconds
    if args.sim:
        from vexo_open.audio import ogg_packets
        sample = KIT / "dashboard" / "speech" / "sample-speech.opus"
        packets = ogg_packets(sample)
        connect_options["audio_packets"] = packets
        seconds = min(seconds, len(packets) / 50)
        print("Using simulator with the kit's public-domain voice.")
    else:
        print("Close the Vexo dashboard tab first, double-tap band 79, then connect.")
    if args.text:
        print("Privacy: audio will be sent to Deepgram. Obtain consent from people nearby.")

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = ROOT / "data" / session_id
    folder.mkdir(parents=True, exist_ok=False)
    audio_path = folder / "audio.ogg"
    json_path = folder / "session.json"

    band = None
    ogg = None
    ears = None
    mic_on = False
    transcript = ""
    final_segments = []
    start_time = None
    capture_error = None
    try:
        band = Band.connect(BAND_NUMBER, transport=transport, **connect_options)
        if band.number != BAND_NUMBER:
            raise RuntimeError("Band identity did not match 79; refusing to record")
        ogg = OggOpusWriter(str(audio_path))
        if args.text:
            ears = speech.create(language=args.language)

            def on_text(text, final):
                if final and text.strip():
                    print("Heard:", text)
                    final_segments.append({
                        "text": text,
                        "received_seconds": round(time.monotonic() - start_time, 2) if start_time else None,
                    })
            ears.start(on_text)

        def on_audio(packet):
            ogg.write(packet.seq, packet.opus)
            if ears:
                ears.feed(packet.opus, packet.seq)

        band.on("audio", on_audio)
        start_time = time.monotonic()
        band.configure(audio=True)
        mic_on = True
        print("Recording %.1f seconds from %s ..." % (seconds, band.name))
        band.wait(seconds)
    except Exception as exc:
        capture_error = str(exc)
    finally:
        if band is not None:
            if mic_on:
                try:
                    band.configure(audio=False)
                except Exception as exc:
                    capture_error = capture_error or str(exc)
            try:
                band.disconnect()
            except Exception:
                pass
        if ogg is not None:
            ogg.close()
        if ears is not None:
            try:
                transcript = ears.stop()
            except Exception as exc:
                capture_error = capture_error or ("Deepgram: " + str(exc))

    if ogg is None:
        try:
            folder.rmdir()
        except OSError:
            pass
        raise RuntimeError(capture_error or "Could not start recording")

    session = {
        "session_id": session_id,
        "band_number": BAND_NUMBER,
        "device": "Vexo-HACK-79" if not args.sim else "Vexo-HACK-79 (simulator)",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": transport,
        "audio_file": "audio.ogg",
        "duration_seconds": round(ogg.seconds, 3),
        "language": args.language if args.text else None,
        "transcription": transcript,
        "transcript_segments": final_segments,
        "packet_loss": ogg.lost,
        "pain_score_self_report_0_to_10": None,
        "note": "Received_seconds is a receive-time estimate, not exact word alignment.",
        "warning": capture_error,
    }
    json_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Audio:", audio_path)
    print("Metadata:", json_path)
    if capture_error:
        print("Warning:", capture_error, file=sys.stderr)
    if ogg.frames == 0:
        raise RuntimeError("No audio packets recorded; see warning above")
    if capture_error:
        raise RuntimeError("Recording saved but incomplete: " + capture_error)
    print("Done.")


if __name__ == "__main__":
    main()
