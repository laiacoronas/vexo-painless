"""Vexo pain-patterns UI: illustrated historical examples + live wearable movement.

Run: python -m streamlit run app/vexo_demo.py
The official Vexo Web Bluetooth library connects in the companion browser tab.
"""
from __future__ import annotations

import html
import io
import os
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from vexo_demo_data import (
    TZ, MOVEMENT_CANDIDATES, MOVEMENT_HISTORY_CANDIDATES, MOTION_SIGNAL_CANDIDATES, VOICE_CANDIDATES,
    CHECKINS, MOMENTS, EXPERIMENT, choose_file, csv_download, demo_movement,
    live_movement, live_voice, live_signal, demo_signal, log_checkin, log_moment, movement_rows,
    read_checkins, read_demo_events, read_json, summaries, weekly_counts,
)

st.set_page_config(page_title="Vexo | Personal patterns", page_icon="⌁", layout="centered", initial_sidebar_state="expanded")

BLUE = "#35658b"
ORANGE = "#e9a046"
RED = "#a53318"
MUTED = "#6e7785"
BG = "#edf0f5"

st.markdown("""
<style>
:root{color-scheme:light;}
.stApp{background:#edf0f5;color:#1b2330;}
.block-container{max-width:760px;padding-top:1.25rem;padding-bottom:5rem;}
[data-testid="stSidebar"]{background:#f8f9fb;}
[data-testid="stSidebar"] *{font-family:Inter,Arial,sans-serif;}
[data-testid="stHeader"]{background:transparent;}
.v-brand{font-size:25px;letter-spacing:-1px;font-weight:850;color:#35658b;line-height:1.2;}
.v-eyebrow{font-size:12px;color:#6d7a87;font-weight:650;letter-spacing:.7px;text-transform:uppercase;}
.v-title{font-size:34px;line-height:1.13;font-weight:790;letter-spacing:-1.1px;color:#17202b;margin:8px 0 5px;}
.v-sub{font-size:14px;color:#76808e;line-height:1.6;margin:3px 0 18px;}
.v-top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;}
.v-pill{display:inline-block;padding:6px 12px;border-radius:30px;background:#dce8f6;color:#285475;font-size:12px;font-weight:800;white-space:nowrap;}
.v-pill.orange{background:#fbe5c9;color:#97530f;}.v-pill.red{background:#f5d1c6;color:#922d17;}
.v-pill.grey{background:#e9edf2;color:#6e7785;}
.v-card{background:#fff;border:1px solid #e3e7ec;border-radius:22px;padding:21px 22px;margin:12px 0;box-shadow:0 5px 18px rgba(27,46,70,.035);}
.v-row{display:flex;justify-content:space-between;align-items:center;gap:12px;}
.v-card-title{font-size:18px;font-weight:760;color:#1e2733;}
.v-muted{color:#78818e;font-size:13px;line-height:1.55;}
.v-big{font-size:31px;line-height:1.1;font-weight:770;letter-spacing:-.8px;}
.v-metric-label{font-size:11px;color:#6b7482;text-transform:uppercase;letter-spacing:.6px;font-weight:700;}
.v-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:17px;}
.v-metric{border-radius:12px;background:#f5f7fa;padding:12px 10px;min-width:0;}
.v-metric-val{font-size:18px;font-weight:780;line-height:1.3;}
.v-meter{position:relative;height:15px;border-radius:10px;overflow:visible;display:flex;gap:3px;margin:25px 0 8px;}
.v-meter .s1{width:35%;border-radius:10px 0 0 10px;background:#dce8f5;}
.v-meter .s2{width:20%;background:#fae3c3;}
.v-meter .s3{width:45%;border-radius:0 10px 10px 0;background:#f4d3c9;}
.v-marker{position:absolute;top:-6px;width:24px;height:24px;border:3px solid white;border-radius:100%;background:#152439;box-shadow:0 1px 6px #8c9aa5;transform:translateX(-50%);}
.v-mark-labels{display:flex;justify-content:space-between;font-size:11px;font-weight:700;}
.v-warn{padding:15px;border-radius:15px;background:#edf3fa;color:#365977;font-size:13px;line-height:1.55;margin:12px 0;}
.v-grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.v-stat-card{background:#fff;border-radius:18px;border:1px solid #e3e7ec;padding:18px;}
.v-stat{font-size:32px;font-weight:760;letter-spacing:-.7px;}
.v-foot{font-size:12px;color:#7b8491;line-height:1.55;margin-top:15px;}
.v-tiny{font-size:11px;color:#7d8896;}
.v-square{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;}
.v-inline-kpi{display:inline-block;padding:7px 11px;border-radius:15px;background:#fbe5c8;color:#85511f;font-size:12px;font-weight:750;margin-right:6px;}
.v-inline-kpi.red{background:#f4d6cb;color:#9f361d;}
.v-separator{height:1px;background:#e8edf0;margin:14px 0;}
.stButton>button[kind="primary"]{background:#34648b;border:0;border-radius:12px;color:#fff;font-weight:720;}
.stButton>button{border-radius:12px;}
.stDownloadButton>button{border-radius:12px;font-weight:680;}
.stRadio [role="radiogroup"]{gap:6px;flex-wrap:wrap;}
.stRadio [role="radiogroup"] label{background:#fff;border:1px solid #e3e7ed;padding:6px 10px;border-radius:12px;font-size:13px;}
hr{border-color:#e2e6ed;}
@media(max-width:600px){
.block-container{padding-left:.8rem;padding-right:.8rem;}
.v-card{padding:16px 15px;}.v-title{font-size:29px;}.v-big{font-size:26px;}.v-metric-val{font-size:14px;}
.v-grid2{gap:8px;}.v-stat-card{padding:12px;}
}
</style>
""", unsafe_allow_html=True)


def e(value):
    return html.escape(str(value))


def card(text):
    st.markdown(f'<div class="v-card">{text}</div>', unsafe_allow_html=True)


def pill(label, tone="blue"):
    return f'<span class="v-pill {tone}">{e(label)}</span>'


def title(text, description=""):
    st.markdown(f'<div class="v-title">{e(text)}</div><div class="v-sub">{e(description)}</div>', unsafe_allow_html=True)


def mode_label():
    return "PREVIEW" if st.session_state.monitor_mode == "Preview" else "LIVE DEVICE"


def current_snapshot():
    if st.session_state.monitor_mode == "Preview":
        payload = demo_movement(st.session_state.demo_scenario)
        return payload, "connected", 0., "preview"
    path = choose_file(MOVEMENT_CANDIDATES, "VEXO_MOVEMENT_JSON")
    payload, state, age = live_movement(path, require_hardware=True)
    return payload, state, age, "live"


def status_display(payload, state):
    if state == "connected" and payload:
        labels = {"typical": ("Typical movement", "blue"),
                  "change": ("Movement changed", "orange"),
                  "strong_change": ("Sustained change", "red")}
        return labels.get(payload.get("movement_status"), ("Unavailable", "grey"))
    mapping = {"waiting":"Waiting for bracelet", "stale":"Stale data",
               "disconnected":"Disconnected", "simulated_source":"No live sensor reading",
               "insufficient_data":"Insufficient data", "calibrating":"Calibrating",
               "malformed":"Invalid stream"}
    return mapping.get(state, "Unavailable"), "grey"


def meter(index, label, kind="movement"):
    marker = ""
    if index is not None:
        idx = max(2., min(98., float(index)))
        marker = f'<div class="v-marker" style="left:{idx:.2f}%"></div>'
    labels = ("Typical", "Changed", "Sustained") if kind == "movement" else ("Normal", "Moderate", "High")
    return (f'<div class="v-meter"><span class="s1"></span><span class="s2"></span>'
            f'<span class="s3"></span>{marker}</div>'
            f'<div class="v-mark-labels"><span style="color:#456c90">{labels[0]}</span>'
            f'<span style="color:#a5682b">{labels[1]}</span><span style="color:#a54831">{labels[2]}</span></div>')


def fmt(value, decimals=1, suffix=""):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return "—"


def movement_card(payload, state, source):
    label, tone = status_display(payload, state)
    idx = (payload or {}).get("movement_deviation_index") if state == "connected" else None
    feat = (payload or {}).get("features") or {}
    activity = (payload or {}).get("activity_state") or "unknown"
    if source == "preview":
        annotation = "Example movement profile. Connect your bracelet to view live measurements."
    elif state != "connected":
        annotation = "No valid live assessment. The UI will recover automatically when fresh readings arrive."
    else:
        annotation = f"Compared with your personal baseline for {activity}. Sensor-derived change, not a pain score."
    index_text = f'{float(idx):.1f}<span style="font-size:15px;color:#8994a0"> / 100</span>' if idx is not None else "—"
    step_status = (payload or {}).get("steps_status")
    step_text = "—" if step_status in {"counter_reset", "counter_jump", "initializing", "no_update", "disconnected"} else fmt(feat.get("steps_in_window"), 0)
    card(f'''
      <div class="v-row"><div class="v-card-title">Movement</div>{pill(label,tone)}</div>
      <div class="v-row" style="margin-top:10px"><div class="v-big">{index_text}</div>
      <span class="v-tiny">Activity: {e(activity)}</span></div>
      {meter(idx,label)}
      <div class="v-metrics">
        <div class="v-metric"><div class="v-metric-label">Accel variation</div><div class="v-metric-val">{fmt(feat.get('accel_std_g'),3,' g')}</div></div>
        <div class="v-metric"><div class="v-metric-label">Wrist rotation</div><div class="v-metric-val">{fmt(feat.get('gyro_rms_dps'),1,'°/s')}</div></div>
        <div class="v-metric"><div class="v-metric-label">Steps / {fmt((payload or {}).get('window_seconds'),0)}s</div><div class="v-metric-val">{step_text}</div></div>
      </div><div class="v-muted" style="margin-top:12px">{e(annotation)}</div>
      {'<div class="v-tiny">Step counter anomaly detected: value withheld until resynchronized.</div>' if step_status in {"counter_reset", "counter_jump"} else ""}
    ''')


def voice_card():
    voice_url = e(os.environ.get("VEXO_VOICE_URL", "http://localhost:8766"))
    conversation_link = f'<a href="{voice_url}" target="_blank" rel="noopener" style="display:inline-block;margin-top:14px;color:#35658b">Open conversation</a>'
    voice_path = choose_file(VOICE_CANDIDATES, "VEXO_VOICE_JSON")
    voice, status = live_voice(voice_path)
    if status != "ready":
        card(f'''
        <div class="v-row"><div class="v-card-title">Voice</div>{pill('Model pending','grey')}</div>
        <div style="margin-top:12px" class="v-muted">Start a conversation to connect the voice model.</div>
        <div style="height:14px;border-radius:10px;background:#eef0f4;margin:20px 0 4px"></div>
        <div class="v-tiny">Pain prediction · optional</div>
        {conversation_link}
        ''')
        return
    label = voice["level"].capitalize()
    tone = {"normal":"blue", "moderate":"orange", "high":"red"}[voice["level"]]
    card(f'''
        <div class="v-row"><div class="v-card-title">Voice</div>{pill(label,tone)}</div>
        <div class="v-muted" style="margin-top:10px">Voice-based pain estimate</div>
        {('<div class="v-stat">' + e(f"{voice['score']:.1f}") + '<span class="v-tiny"> / 10</span></div>') if voice.get('score') is not None else ''}
        {meter(voice['index'],label,kind='voice') if voice['index'] is not None else ''}
        {conversation_link}
    ''')


def plot_raw_imu(source):
    """Real IMU trace is not a pain estimate and is separate from 5s assessments."""
    if source == "preview":
        points = demo_signal(st.session_state.demo_scenario)
        signal_state = "preview"
    else:
        signal_path = choose_file(MOTION_SIGNAL_CANDIDATES, "VEXO_MOTION_SIGNAL_JSON")
        points, signal_state = live_signal(signal_path)
    st.markdown('<div class="v-card-title" style="margin:20px 0 2px">Live wrist motion</div>', unsafe_allow_html=True)
    st.caption("25 Hz device sampling · ~10 Hz graph data · ~1 update/s. Movement ≠ pain.")
    if not points:
        st.info("Waiting for fresh motion samples. The 5-second index is a separate assessment.")
        return
    fig=go.Figure()
    times=[x["timestamp"] for x in points]
    fig.add_trace(go.Scatter(x=times,y=[x["accel"] for x in points],name="Acceleration magnitude (g)",
                             mode="lines",line=dict(color=BLUE,width=2),hovertemplate="%{y:.3f} g<extra></extra>"))
    fig.add_trace(go.Scatter(x=times,y=[x["gyro"] for x in points],name="Angular speed (°/s)",
                             mode="lines",yaxis="y2",line=dict(color=ORANGE,width=2),
                             hovertemplate="%{y:.1f} °/s<extra></extra>"))
    now=datetime.now(TZ)
    fig.update_layout(height=255,margin=dict(l=8,r=12,t=8,b=8),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[now-timedelta(seconds=22),now],tickformat="%H:%M:%S",showgrid=False),
        yaxis=dict(title="g",gridcolor="#e6ebf0",zeroline=False,side="left"),
        yaxis2=dict(title="°/s",overlaying="y",side="right",showgrid=False,zeroline=False),
        legend=dict(orientation="h",y=-.23,x=0,font=dict(size=10)),
        font=dict(family="Arial",color="#687483",size=10))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"live_imu_{source}")
    if signal_state=="preview":
        st.caption("Example motion trace · Connect the band to see your own readings.")


def plot_live_trend(payload, state, source):
    fig = go.Figure()
    if source == "preview":
        base = (payload or {}).get("movement_deviation_index") or 17
        now = datetime.now(TZ)
        times = [now - timedelta(seconds=5 * (19-i)) for i in range(20)]
        values = [max(0,min(100,base+7*math.sin(i*.48) - 3*math.cos(i*.27))) for i in range(20)]
    else:
        history_path = choose_file(MOVEMENT_HISTORY_CANDIDATES,"VEXO_MOVEMENT_CSV")
        valid = movement_rows(history_path)
        times = [r["timestamp"] for r in valid[-35:]]
        values = [r["index"] for r in valid[-35:]]
    if not values:
        st.info("Waiting for valid movement windows to build the live trend.")
        return
    fig.add_hrect(y0=0,y1=35,fillcolor="#dbe8f4",line_width=0,opacity=.5)
    fig.add_hrect(y0=35,y1=55,fillcolor="#fbe8cc",line_width=0,opacity=.35)
    fig.add_hrect(y0=55,y1=100,fillcolor="#f3d4c9",line_width=0,opacity=.24)
    fig.add_trace(go.Scatter(x=times,y=values, mode="lines+markers", name="Movement deviation",
                             line=dict(color=BLUE,width=3),marker=dict(color=BLUE,size=5),
                             hovertemplate="%{x|%H:%M:%S}<br>Deviation %{y:.1f}<extra></extra>"))
    fig.update_layout(margin=dict(l=4,r=7,t=8,b=5),height=195,
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,yaxis=dict(range=[0,102],tickvals=[0,35,55,100],gridcolor="#e5e9ef",zeroline=False),
        xaxis=dict(showgrid=False,tickformat="%H:%M:%S",tickfont=dict(size=10)),
        font=dict(family="Arial",color="#667485",size=11))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"trend_{source}")


def checkin_panel(payload, state):
    st.markdown("<div class='v-card-title' style='margin:22px 0 4px'>How are you feeling?</div>",unsafe_allow_html=True)
    st.caption("Pain is **self-reported** (0–10). This is separate from the movement-deviation index and is saved for later evaluation/personalization.")
    with st.form("vexo_checkin_form", clear_on_submit=False):
        pain = st.slider("Self-reported pain",0,10,4,1,help="0 = no pain, 10 = worst pain imaginable")
        col_a,col_b = st.columns(2)
        with col_a: activity = st.selectbox("What were you doing?",["Sitting","Standing up","Walking","Exercising","Resting","Other"])
        with col_b: body_area = st.selectbox("Location",["Not specified","Lower back","Knee","Hip","Shoulder","Head","Other"])
        note = st.text_input("Optional note",placeholder="e.g. getting up after a long meeting")
        submit = st.form_submit_button("Save pain check-in",type="primary",use_container_width=True)
    if submit:
        try:
            log_checkin(pain,activity,body_area,note,payload,state)
            st.success("Saved locally with its timestamp and corresponding movement window.")
        except OSError as exc:
            st.error(f"Could not save the check-in: {exc}")
    checkins = read_checkins()
    if checkins:
        last = checkins[-1]
        st.caption(f"{len(checkins)} saved check-in(s) · Last reported pain: {last['self_reported_pain_0_10']}/10")


def latest_panel():
    payload,state,age,source = current_snapshot()
    label,tone=status_display(payload,state)
    pill_text = "PREVIEW" if source == "preview" else ("LIVE · VEXO 79" if state == "connected" else "NOT CONNECTED")
    pill_tone = "orange" if source=="preview" else ("blue" if state=="connected" else "grey")
    st.markdown(f'<div class="v-top"><div class="v-brand">vexo</div>{pill(pill_text,pill_tone)}</div>',unsafe_allow_html=True)
    title("Right now","Continuous movement monitoring. Motion changes are not a pain measurement; voice will be integrated separately.")
    if source=="preview":
        card(f'<div class="v-row">{pill(label,tone)}<span class="v-muted">Illustrative stream — no physical sensor required.</span></div>')
    elif state=="connected":
        card(f'<div class="v-row">{pill(label,tone)}<span class="v-muted">Fresh Vexo data · {age:.0f}s ago</span></div>')
    else:
        card(f'<div class="v-row">{pill(label,tone)}<span class="v-muted">Connect Vexo 79 through localhost:8765/ble_bridge/index.html.</span></div>')
    movement_card(payload,state,source)
    plot_raw_imu(source)
    voice_card()
    st.markdown("<div class='v-card-title' style='margin:20px 0 1px'>Movement · last windows</div>",unsafe_allow_html=True)
    plot_live_trend(payload,state,source)
    if state == "connected" and payload and payload.get("movement_status") in ("change","strong_change"):
        st.markdown('<div class="v-warn"><strong>Change detected.</strong> This movement pattern differs from your baseline. Consider a voluntary check-in; movement alone does not establish that you are experiencing pain.</div>',unsafe_allow_html=True)
    st.markdown('<div class="v-warn">ⓘ These categories describe <strong>movement deviation only</strong>. Pain is recorded separately from the user; no automatic pain diagnosis is being made.</div>',unsafe_allow_html=True)
    st.markdown('<div class="v-foot">Personal wellness prototype · Not a medical diagnosis. Example history is illustrative and is separate from live wearable readings.</div>',unsafe_allow_html=True)
    with st.expander("Connection diagnostics / integration paths"):
        st.write("Monitoring view:",mode_label())
        st.code(str(choose_file(MOVEMENT_CANDIDATES,"VEXO_MOVEMENT_JSON")),language=None)
        st.write("Status:",state,"| Age (s):",age)
        st.write("Data origin:","example" if source=="preview" else "wearable", "| Samples:",(payload or {}).get("sample_count"))
        st.write("Steps counter (raw):",(payload or {}).get("steps_total"),"| Counter quality:",(payload or {}).get("steps_status"))
        st.code(str(choose_file(MOTION_SIGNAL_CANDIDATES,"VEXO_MOTION_SIGNAL_JSON")), language=None)
        st.write("Voice adapter path:",str(choose_file(VOICE_CANDIDATES,"VEXO_VOICE_JSON")))


def timeline_plot(events, live_rows):
    as_of = datetime.now(TZ).date()
    first = as_of - timedelta(days=13)
    fig = go.Figure()
    for i in range(14):
        day = first + timedelta(days=i)
        if day.weekday() in (0,3):
            fig.add_hrect(y0=i-.42,y1=i+.42,fillcolor="#e8edf3",opacity=.8,line_width=0)
        fig.add_shape(type="line",x0=6,x1=22,y0=i,y1=i,line=dict(color="#d2d8e0",width=1.5),layer="below")
    for severity,color,marker in (("moderate",ORANGE,11),("high",RED,15)):
        items = [r for r in events if r["severity"]==severity]
        fig.add_trace(go.Scatter(
            x=[r["hour_float"] for r in items],y=[r["day_index"] for r in items],
            mode="markers",name=f"Example {severity}",
            marker=dict(color=color,size=marker,line=dict(color="#fff",width=1.4)),
            customdata=[r["context"] for r in items],
            hovertemplate="%{customdata}<br>%{x:.1f} h · example record<extra></extra>",
        ))
    # Actual Vexo events, visually distinguished and never relabeled as historical pain events.
    flagged = [r for r in live_rows if r["status"] in ("change","strong_change") and first <= r["timestamp"].date() <= as_of]
    if flagged:
        fig.add_trace(go.Scatter(x=[r["timestamp"].hour+r["timestamp"].minute/60 for r in flagged],
                                 y=[(r["timestamp"].date()-first).days for r in flagged],
                                 mode="markers",name="Real movement flags",
                                 marker=dict(color="#35658b",symbol="diamond",size=9,line=dict(color="white",width=1)),
                                 hovertemplate="Real movement flag at %{x:.1f}h<extra></extra>"))
    yticks = [first + timedelta(days=i) for i in range(14)]
    # Use portable day formatting on Windows.
    labels = ["Today" if d==as_of else f"{d.strftime('%a')} {d.day}" for d in yticks]
    fig.update_layout(height=525,margin=dict(l=4,r=6,t=20,b=12),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#627284",size=11),
                      legend=dict(orientation="h",y=-.09,x=0,font=dict(size=10)),
                      xaxis=dict(range=[5.5,22.5],tickvals=[6,10,14,18,22],ticktext=["6a","10a","2p","6p","10p"],side="top",showgrid=False,zeroline=False),
                      yaxis=dict(range=[13.7,-.7],tickvals=list(range(14)),ticktext=labels,showgrid=False,zeroline=False))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="timeline")


def timeline_panel(events):
    st.markdown('<div class="v-top"><div class="v-brand">vexo</div>'+pill('EXAMPLE HISTORY','grey')+'</div>',unsafe_allow_html=True)
    title("Last 14 days","Illustrative 14-day history. Real movement observations are shown separately as blue diamonds when available.")
    stats = summaries(events)
    st.markdown(f'<div style="margin:4px 0 16px"><span class="v-inline-kpi">● {stats["moderate"]} moderate</span><span class="v-inline-kpi red">● {stats["high"]} high</span></div>',unsafe_allow_html=True)
    recent = movement_rows(choose_file(MOVEMENT_HISTORY_CANDIDATES,"VEXO_MOVEMENT_CSV"),recent_hours=14*24)
    card('<div class="v-card-title">Flagged moments</div><div class="v-muted">Rows shaded on Mondays and Thursdays; diamonds denote real Vexo movement changes, if available.</div>')
    timeline_plot(events,recent)
    card(f'<div class="v-card-title">A pattern to explore</div><div class="v-muted" style="margin-top:8px"><b>{stats["mon_thu"]} of {stats["high"]} example high moments</b> fall on Mondays and Thursdays. This is demo storytelling, not an observation about you.</div>')
    if recent:
        st.caption(f"Separate real movement record: {len(recent)} valid windows in the last 14 days. These are not automatically labeled as pain events.")


def pattern_panel(events):
    st.markdown('<div class="v-top"><div class="v-brand">vexo</div>'+pill('EXAMPLE PATTERNS','grey')+'</div>',unsafe_allow_html=True)
    stats=summaries(events)
    title("Your patterns",f"From {len(events)} example moments across 14 days. Associations, not causes.")
    total=max(1,stats["high"])
    card(f'<div class="v-row"><div class="v-card-title">Long sitting</div>{pill("Example · lower back", "grey")}</div>'
         f'<div class="v-muted" style="margin:12px 0"><b>{stats["long_sitting"]} of {stats["high"]}</b> example high moments came after 2+ hours of sitting.</div>'
         '<div style="display:flex;gap:4px">'+''.join(f'<div style="height:12px;flex:1;background:{RED if i<stats["long_sitting"] else "#e7ebf0"};border-radius:3px"></div>' for i in range(stats["high"]))+'</div>')
    card(f'<div class="v-card-title">Mondays and Thursdays</div><div class="v-muted" style="margin-top:9px"><b>{stats["mon_thu"]} of {stats["high"]}</b> example high moments fell on these days.</div>')
    card(f'<div class="v-card-title">Mornings</div><div class="v-muted" style="margin-top:9px"><b>{stats["morning"]} of {stats["moderate"]}</b> example moderate moments happened between 7–9 AM.</div>')
    with_walk_days={e["day"] for e in events if e["had_lunch_walk"]}
    all_days={e["day"] for e in events}
    without=all_days-with_walk_days
    walk_flags=sum(e["day"] in with_walk_days for e in events)
    no_flags=sum(e["day"] in without for e in events)
    w_rate=walk_flags/len(with_walk_days) if with_walk_days else 0
    n_rate=no_flags/len(without) if without else 0
    card(f'<div class="v-card-title">Lunch walks</div><div class="v-muted" style="margin-top:8px">Flagged example moments per day</div>'
         f'<div class="v-row" style="margin-top:12px"><b>With a walk</b><span>{w_rate:.1f} / day</span></div>'
         f'<div class="v-row" style="margin-top:8px"><b>Without</b><span>{n_rate:.1f} / day</span></div>')
    st.markdown('<div class="v-warn">These patterns illustrate what the app could discover over time; they are not conclusions from your current sensor recordings.</div>',unsafe_allow_html=True)


def experiment_panel(events):
    st.markdown('<div class="v-brand">vexo</div>',unsafe_allow_html=True)
    title("Try this week","A voluntary experiment inspired by the example pattern, not clinical advice.")
    card('<div class="v-eyebrow">EXPERIMENT · 7 DAYS</div>'
         '<div style="font-size:25px;font-weight:780;margin:13px 0">Move for 5 minutes for every hour you sit</div>'
         '<div class="v-muted">In the example history, prolonged sitting preceded several flagged moments. You can compare your own patterns over time.</div>'
         '<div class="v-separator"></div>'
         '<div style="margin-bottom:12px">① &nbsp; Record a 55-minute sitting period.</div>'
         '<div style="margin-bottom:12px">② &nbsp; Stand, walk, or stretch for 5 minutes.</div>'
         '<div>③ &nbsp; Compare your self-reports and movement trends after 7 days.</div>')
    state=read_json(EXPERIMENT)
    if state:
        st.success(f"Local experiment started: {state.get('started_at_local','unknown')}. No bracelet vibration is implemented in this demo.")
    else:
        if st.button("Start local experiment tracking",type="primary",use_container_width=True):
            EXPERIMENT.parent.mkdir(parents=True,exist_ok=True)
            state={"started_at_local":datetime.now(TZ).isoformat(),"duration_days":7,
                   "title":"5-minute movement breaks per sitting hour","mode":"demonstration"}
            try:
                EXPERIMENT.write_text(__import__('json').dumps(state,indent=2),encoding="utf-8")
                st.success("Started. Stored locally; no automatic haptic reminders.")
            except OSError as exc:
                st.error(f"Could not save experiment: {exc}")
    st.caption("No vibration is sent to the bracelet; haptic reminders would require separate device integration.")


def moment_panel():
    payload,state,age,source=current_snapshot()
    st.markdown('<div class="v-top"><div class="v-brand">vexo</div>'+pill('PREVIEW' if source=='preview' else 'LIVE SENSOR' if state=='connected' else 'OFFLINE', 'orange' if source=='preview' else 'blue' if state=='connected' else 'grey')+'</div>',unsafe_allow_html=True)
    title("Live moment","A movement-change check-in. This does not establish that pain is present.")
    label,tone=status_display(payload,state)
    card(f'<div class="v-row">{pill(label,tone)}<span class="v-muted">Latest valid movement window</span></div>'
         f'<div class="v-muted" style="margin-top:12px">A voluntary check-in helps evaluate whether this sensor change is related to pain for this user.</div>')
    movement_card(payload,state,source)
    if state=="connected":
        context=st.text_area("What were you doing?",placeholder="e.g. getting up from my chair",max_chars=200,key="moment_context")
        col1,col2=st.columns(2)
        with col1:
            if st.button("False alarm",use_container_width=True):
                try:log_moment("false_alarm",payload,state,context);st.success("False alarm recorded locally.")
                except OSError as exc:st.error(str(exc))
        with col2:
            if st.button("Save moment",type="primary",use_container_width=True):
                try:log_moment("saved_moment",payload,state,context);st.success("Moment saved. Add a pain check-in if desired.")
                except OSError as exc:st.error(str(exc))
        checkin_panel(payload,state)
    else:
        st.info("No fresh sensor data. Switch to Preview to explore this screen, or reconnect Bluetooth.")
    st.caption("Speech transcription is available through the Voice card.")


def make_pdf_report(events, checkins):
    """Generate a local report with clearly marked example data."""
    try:
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.colors import HexColor
    except ImportError:
        return None
    out=io.BytesIO(); pdf=Canvas(out,pagesize=(595,842));
    pdf.setFillColor(HexColor("#35658b"));pdf.setFont("Helvetica-Bold",22);pdf.drawString(55,785,"vexo | Weekly summary")
    pdf.setFillColor(HexColor("#424e5e"));pdf.setFont("Helvetica",12)
    pdf.drawString(55,750,"ILLUSTRATIVE HISTORICAL EXAMPLE - not actual wearable data")
    days,counts=weekly_counts(events); mod=sum(counts[d]["moderate"] for d in days); high=sum(counts[d]["high"] for d in days)
    pdf.setFont("Helvetica-Bold",15);pdf.drawString(55,705,f"{mod} moderate moments     {high} high moments")
    y=665
    pdf.setFont("Helvetica",11)
    for day in days:
        pdf.drawString(55,y,f"{day.strftime('%a %d %b')}:  {counts[day]['moderate']} moderate; {counts[day]['high']} high")
        y-=24
    y-=25
    pdf.setFont("Helvetica-Bold",13);pdf.drawString(55,y,"Example patterns");y-=24
    stats=summaries(events);pdf.setFont("Helvetica",11)
    for row in [f"{stats['long_sitting']} of {stats['high']} high moments followed long sitting.",
                f"{stats['mon_thu']} of {stats['high']} high moments occurred on Mon/Thu.",
                f"{stats['morning']} of {stats['moderate']} moderate moments were 7-9 AM.",
                f"User-entered check-ins stored locally: {len(checkins)}"]:
        pdf.drawString(55,y,row);y-=22
    pdf.setFont("Helvetica",9);pdf.setFillColor(HexColor("#65707e"))
    pdf.drawString(55,96,"Movement deviations are not a pain measurement or diagnosis.")
    pdf.drawString(55,78,"No data have been transmitted or shared by generating this PDF.")
    pdf.save();return out.getvalue()


def share_panel(events):
    st.markdown('<div class="v-top"><div class="v-brand">vexo</div>'+pill('EXAMPLE REPORT','grey')+'</div>',unsafe_allow_html=True)
    title("Share","A sample weekly summary. Nothing is shared until you explicitly download a file.")
    days,counts=weekly_counts(events)
    mod=sum(counts[d]["moderate"] for d in days)
    high=sum(counts[d]["high"] for d in days)
    st.markdown(f'<div class="v-grid2"><div class="v-stat-card"><div class="v-metric-label">Moderate · example</div><div class="v-stat" style="color:#a1651e">{mod}</div></div><div class="v-stat-card"><div class="v-metric-label">High · example</div><div class="v-stat" style="color:#a53318">{high}</div></div></div>',unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=[d.strftime("%a") for d in days],y=[counts[d]["moderate"] for d in days],name="Moderate · example",marker_color=ORANGE))
    fig.add_trace(go.Bar(x=[d.strftime("%a") for d in days],y=[counts[d]["high"] for d in days],name="High · example",marker_color=RED))
    fig.update_layout(barmode="stack",height=250,margin=dict(l=4,r=4,t=20,b=2),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h",y=-.2),yaxis=dict(showgrid=True,gridcolor="#e8edf2",zeroline=False),xaxis=dict(showgrid=False))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    stats=summaries(events)
    card(f'<div class="v-card-title">Top example patterns</div><div class="v-muted" style="margin-top:10px">• Long sitting preceded {stats["long_sitting"]}/{stats["high"]} example high moments.<br>• Most of these examples occurred on Monday/Thursday.<br>• Frequently mentioned body area: lower back.</div>')
    checkins=read_checkins()
    report=make_pdf_report(events,checkins)
    if report:
        st.download_button("Download example weekly report (PDF)",data=report,
                           file_name="vexo_example_weekly_summary.pdf",mime="application/pdf",use_container_width=True,type="primary")
    else:
        st.info("Install reportlab to enable the PDF download (pip install reportlab).")
    st.download_button("Download your check-ins (CSV)",data=csv_download(checkins),
                       file_name="vexo_self_reports.csv",mime="text/csv",use_container_width=True)
    st.caption("The weekly report shows illustrative history, not actual user recordings. The check-in CSV contains your locally entered responses.")


# Sidebar is intentionally simple for reliable presentation switching.
with st.sidebar:
    st.markdown('<div class="v-brand" style="margin:3px 0 8px">vexo</div>',unsafe_allow_html=True)
    st.markdown("### Monitoring")
    st.radio("Monitoring mode",["Live device","Preview"],key="monitor_mode",
             help="Live device shows band measurements. Preview illustrates how the interface works using example values.")
    if st.session_state.monitor_mode=="Preview":
        st.selectbox("Movement scenario",["Typical","Moderate","High"],key="demo_scenario")
        st.caption("Preview displays illustrative values. No bracelet measurement is used in this mode.")
    else:
        st.caption("Open a second terminal and run:")
        st.code("python .\\ble_bridge\\server.py",language="powershell")
        st.markdown("**[Open Vexo 79 connector](http://localhost:8765/ble_bridge/index.html)**")
        st.caption("Connect in Chrome or Edge. Close the original Vexo dashboard first; one Bluetooth connection at a time.")
        st.caption("Live sensor trace refreshes about once per second; baseline assessment updates every five seconds.")
    st.divider()
    st.caption("Open the Voice card to start a conversation.")
    st.caption("All check-ins and flags are stored locally in vexo-pain-monitor/data/.")

navigation=["Now","Timeline","Patterns","Experiment","Moment","Share"]
page=st.radio("Screens",navigation,horizontal=True,label_visibility="collapsed")
st.markdown('<div style="height:5px"></div>',unsafe_allow_html=True)

# Streamlit fragments refresh independently; no browser-side JavaScript or network needed.
@st.fragment(run_every="1s")
def now_refresh():
    latest_panel()

@st.fragment(run_every="3s")
def moment_refresh():
    moment_panel()

if page=="Now":
    now_refresh()
    _payload,_state,_,_ = current_snapshot()
    checkin_panel(_payload,_state)
elif page=="Moment":
    moment_refresh()
else:
    history=read_demo_events()
    if page=="Timeline":timeline_panel(history)
    elif page=="Patterns":pattern_panel(history)
    elif page=="Experiment":experiment_panel(history)
    elif page=="Share":share_panel(history)

st.markdown('<div class="v-foot" style="text-align:center;padding:20px 0 4px">VEXO PERSONAL PATTERNS · HACKATHON PROTOTYPE · NOT FOR MEDICAL DECISIONS</div>',unsafe_allow_html=True)
