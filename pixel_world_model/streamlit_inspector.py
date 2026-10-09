"""Live cloud inspector. Launched by the serve-* CLI commands."""
import argparse
import sys
from pathlib import Path

# Streamlit executes this file directly rather than as a package module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from pixel_world_model.web_inspector import InspectorSession, LossHistory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('vae', 'dynamics'), required=True)
    parser.add_argument('--logdir')
    args = parser.parse_args()
    import json
    config = json.loads(args.config)
    stage = args.stage
    st.set_page_config(page_title='Pixel world model inspector', layout='wide')
    st.title(f'{stage.capitalize()} · live inspection')
    st.caption('Held-out predictions from the latest saved model. Training and validation share the same epoch axis.')
    live = st.sidebar.toggle('Live updates', value=True)
    interval = st.sidebar.slider('Refresh interval (seconds)', 1, 30, 5)
    st.sidebar.code(config[f'{stage}_checkpoint'])
    st.sidebar.caption(f"Requested device: {config['device']}")
    identity = (args.config, args.stage, args.logdir)
    if st.session_state.get('identity') != identity:
        st.session_state.clear()
        st.session_state.identity = identity
        st.session_state.history = None
        st.session_state.inspector = None

    @st.fragment(run_every=interval if live else None)
    def dashboard():
        try:
            if st.session_state.history is None:
                st.session_state.history = LossHistory(config, stage, args.logdir)
            history = st.session_state.history.read()
            rows = [{'Epoch': epoch, 'Loss': loss, 'Series': label}
                    for key, label in [('train', 'Training'), ('validation', 'Validation')]
                    for epoch, loss in history[key]]
            st.subheader('Training and validation objective')
            if rows:
                frame = pd.DataFrame(rows)
                chart = alt.Chart(frame).mark_line(point=True).encode(
                    x=alt.X('Epoch:Q'), y=alt.Y('Loss:Q', scale=alt.Scale(zero=False)),
                    color='Series:N', strokeDash='Series:N',
                    tooltip=['Series:N', 'Epoch:Q', 'Loss:Q']).interactive()
                st.altair_chart(chart, use_container_width=True)
                with st.expander('Curve data'):
                    st.dataframe(frame, hide_index=True)
            else:
                st.info('Waiting for training metrics. Validation appears after the first completed epoch.')
        except Exception as exc:
            st.warning(f'Waiting for training data: {exc}')

        st.subheader('Held-out visual comparison')
        try:
            if st.session_state.inspector is None:
                st.session_state.inspector = InspectorSession(config, stage)
            session = st.session_state.inspector
            state = session.state()
            controls = st.columns(3)
            sample = controls[0].number_input('Held-out sample', min_value=1,
                                             max_value=state['samples'], value=state['sample'] + 1)
            channel = controls[1].selectbox('Stack frame', [1, 2, 3, 4], index=state['channel'])
            horizon = controls[2].number_input('Rollout horizon', 1, 100, state['horizon'], disabled=stage == 'vae')
            if sample - 1 != state['sample']:
                state = session.act({'action': 'sample', 'position': int(sample) - 1})
            if channel - 1 != state['channel']:
                state = session.act({'action': 'channel', 'channel': channel - 1})
            if horizon != state['horizon']:
                state = session.act({'action': 'horizon', 'horizon': int(horizon)})
            if stage == 'dynamics' and state['steps']:
                step = st.slider('Prediction step', 0, state['steps'], min(state['step'], state['steps']))
                if step != state['step']:
                    state = session.act({'action': 'step', 'step': step})
            st.caption(f"Episode {state['episode']} · {state['checkpoint']} · device {session.view.device}")
            for column, panel in zip(st.columns(len(state['panels'])), state['panels']):
                column.image(panel['image'], caption=panel['title'], use_container_width=True)
            st.code(state['metrics'])
            if state['checkpoint_modified']:
                from datetime import datetime
                st.caption(f"Snapshot saved {datetime.fromtimestamp(state['checkpoint_modified']).isoformat(timespec='seconds')}")
        except Exception as exc:
            st.info(f'Waiting for a readable checkpoint: {exc}')

    dashboard()


if __name__ == '__main__':
    main()
