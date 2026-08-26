export interface SegmentOption {
  readonly id: string;
  readonly label: string;
  /** Single-character shortcut shown on the chip. */
  readonly key: string;
}

export interface TuneField {
  readonly key: string;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly value: number;
  readonly format?: (value: number) => string;
}

export interface TuneGroup {
  readonly title: string;
  readonly fields: readonly TuneField[];
}

export interface ModelChoice {
  readonly label: string;
  readonly options: readonly string[];
  readonly value: string;
}

export interface Scenario {
  readonly id: string;
  readonly label: string;
  readonly note: string;
}

export interface HudConfig {
  readonly scenarios: readonly Scenario[];
  readonly rotation: ModelChoice;
  readonly mirror: ModelChoice;
  readonly scenes: readonly SegmentOption[];
  readonly views: readonly SegmentOption[];
  /** Water or smoke. The spout carries one at a time. */
  readonly media: readonly SegmentOption[];
  readonly model: ModelChoice;
  readonly groups: readonly TuneGroup[];
}

export interface HudActions {
  readonly onScene: (id: string) => void;
  readonly onView: (index: number) => void;
  readonly onModel: (id: string) => void;
  readonly onPour: (on: boolean) => void;
  readonly onStorm: (on: boolean) => void;
  readonly onMedium: (id: string) => void;
  readonly onOpen: () => void;
  readonly onScenario: (id: string) => void;
  readonly onRotation: (value: string) => void;
  readonly onMirror: (value: string) => void;
  readonly onDrain: () => void;
  readonly onReset: () => void;
  readonly onTune: (key: string, value: number) => void;
}

export interface Hud {
  setStatus(tone: 'info' | 'error', text: string): void;
  /** Move the spout marker. `depth` is 0..1 from far to near. */
  setSpout(x: number, y: number, depth: number, live: boolean): void;
  /** 0..1 of the water supply currently in the scene. */
  setFill(fraction: number): void;
  setScene(id: string): void;
  setScenario(id: string): void;
  setPouring(on: boolean): void;
  /** Move a slider to a value the app worked out for itself. */
  setTune(key: string, value: number): void;
  setStorm(on: boolean): void;
  setMedium(id: string): void;
  dismissHint(): void;
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  return node;
}

/** Every HUD element the page owns; HTMLElement covers everything used here. */
function requireElement(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!(node instanceof HTMLElement)) {
    throw new Error(`The page is missing #${id}.`);
  }
  return node;
}

/** A row of chips where exactly one is active. */
function buildSegments(
  host: HTMLElement,
  options: readonly SegmentOption[],
  onPick: (id: string, index: number) => void,
): (id: string) => void {
  const buttons = new Map<string, HTMLButtonElement>();

  options.forEach((option, index) => {
    const button = element('button');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(index === 0));
    button.append(option.label);
    const hint = element('kbd');
    hint.textContent = option.key;
    button.append(hint);
    button.addEventListener('click', () => {
      select(option.id);
      onPick(option.id, index);
    });
    host.append(button);
    buttons.set(option.id, button);
  });

  function select(id: string): void {
    for (const [candidate, button] of buttons) {
      button.setAttribute('aria-pressed', String(candidate === id));
    }
  }

  return select;
}

export function createHud(config: HudConfig, actions: HudActions): Hud {
  const status = requireElement('status');
  const hint = requireElement('hint');
  const meter = requireElement('meter');
  const meterValue = requireElement('meterValue');
  const pourButton = requireElement('pour');
  const stormButton = requireElement('storm');
  const spout = requireElement('spout');
  const drawer = requireElement('tune');

  const selectScene = buildSegments(requireElement('scenes'), config.scenes, (id) =>
    actions.onScene(id),
  );

  // Named setups, so the interesting states are one click rather than a hotkey
  // and four sliders.
  const scenarioHost = requireElement('scenarios');
  const scenarioButtons = new Map<string, HTMLButtonElement>();

  function selectScenario(id: string): void {
    for (const [candidate, button] of scenarioButtons) {
      button.setAttribute('aria-pressed', String(candidate === id));
    }
  }

  for (const scenario of config.scenarios) {
    const button = element('button', 'scenario');
    button.type = 'button';
    button.setAttribute('aria-pressed', 'false');
    const title = document.createTextNode(scenario.label);
    const note = element('small');
    note.textContent = scenario.note;
    button.append(title, note);
    button.addEventListener('click', () => {
      selectScenario(scenario.id);
      actions.onScenario(scenario.id);
    });
    scenarioHost.append(button);
    scenarioButtons.set(scenario.id, button);
  }
  const selectView = buildSegments(requireElement('views'), config.views, (_id, index) =>
    actions.onView(index),
  );

  let pouring = false;

  function setPouring(on: boolean): void {
    pouring = on;
    document.body.dataset.pouring = String(on);
    pourButton.setAttribute('aria-pressed', String(on));
    pourButton.firstChild?.replaceWith(on ? 'Pouring' : 'Hold');
  }

  pourButton.addEventListener('click', () => {
    setPouring(!pouring);
    actions.onPour(pouring);
  });

  let storming = false;
  function setStorm(on: boolean): void {
    storming = on;
    stormButton.setAttribute('aria-pressed', String(on));
  }
  stormButton.addEventListener('click', () => {
    setStorm(!storming);
    actions.onStorm(storming);
  });

  const selectMedium = buildSegments(requireElement('media'), config.media, (id) =>
    actions.onMedium(id),
  );
  requireElement('open').addEventListener('click', actions.onOpen);
  requireElement('drain').addEventListener('click', actions.onDrain);
  requireElement('reset').addEventListener('click', actions.onReset);

  function toggleDrawer(open?: boolean): void {
    const next = open ?? drawer.dataset.open !== 'true';
    drawer.dataset.open = String(next);
  }
  requireElement('tuneToggle').addEventListener('click', () => toggleDrawer());
  requireElement('tuneClose').addEventListener('click', () => toggleDrawer(false));

  // --- tuning drawer ---
  const body = requireElement('tuneBody');

  function addChoice(host: HTMLElement, choice: ModelChoice, onPick: (value: string) => void): void {
    const field = element('label', 'field');
    const label = element('span', 'field-label');
    label.append(choice.label);
    const select = element('select');
    for (const option of choice.options) {
      const item = element('option');
      item.value = option;
      item.textContent = option;
      select.append(item);
    }
    select.value = choice.value;
    select.addEventListener('change', () => onPick(select.value));
    field.append(label, select);
    host.append(field);
  }

  const cameraGroup = element('div', 'group');
  const cameraHeading = element('h3');
  cameraHeading.textContent = 'camera';
  cameraGroup.append(cameraHeading);
  addChoice(cameraGroup, config.rotation, actions.onRotation);
  addChoice(cameraGroup, config.mirror, actions.onMirror);
  body.append(cameraGroup);

  const modelGroup = element('div', 'group');
  const modelHeading = element('h3');
  modelHeading.textContent = 'depth model';
  const modelField = element('label', 'field');
  const modelLabel = element('span', 'field-label');
  modelLabel.append(config.model.label);
  const modelSelect = element('select');
  for (const option of config.model.options) {
    const item = element('option');
    item.value = option;
    item.textContent = option;
    modelSelect.append(item);
  }
  modelSelect.value = config.model.value;
  modelSelect.addEventListener('change', () => actions.onModel(modelSelect.value));
  modelField.append(modelLabel, modelSelect);
  modelGroup.append(modelHeading, modelField);
  body.append(modelGroup);

  /** Sliders by key, so measured values can be shown back on the control. */
  const tuneFields = new Map<string, (value: number) => void>();

  for (const group of config.groups) {
    const section = element('div', 'group');
    const heading = element('h3');
    heading.textContent = group.title;
    section.append(heading);

    for (const field of group.fields) {
      const wrapper = element('label', 'field');
      const label = element('span', 'field-label');
      const readout = element('b');
      const show = (value: number) => {
        readout.textContent = field.format ? field.format(value) : value.toFixed(3);
      };

      const input = element('input');
      input.type = 'range';
      input.min = String(field.min);
      input.max = String(field.max);
      input.step = String(field.step);
      input.value = String(field.value);
      input.addEventListener('input', () => {
        const value = Number(input.value);
        show(value);
        actions.onTune(field.key, value);
      });

      show(field.value);
      tuneFields.set(field.key, (value) => {
        input.value = String(value);
        show(value);
      });
      label.append(field.label, readout);
      wrapper.append(label, input);
      section.append(wrapper);
    }
    body.append(section);
  }

  // --- keyboard ---
  addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    const viewIndex = Number(event.key) - 1;
    if (viewIndex >= 0 && viewIndex < config.views.length) {
      const view = config.views[viewIndex];
      if (view) {
        selectView(view.id);
        actions.onView(viewIndex);
      }
      return;
    }
    if (event.code === 'Space') {
      event.preventDefault();
      setPouring(!pouring);
      actions.onPour(pouring);
      return;
    }
    const pressed = event.key.toLowerCase();
    const scene = config.scenes.find((candidate) => candidate.key.toLowerCase() === pressed);
    if (scene) {
      selectScene(scene.id);
      actions.onScene(scene.id);
      return;
    }
    if (pressed === 'o') {
      actions.onOpen();
    } else if (pressed === 's') {
      setStorm(!storming);
      actions.onStorm(storming);
    } else if (pressed === 'd') {
      actions.onDrain();
    } else if (pressed === 'r') {
      actions.onReset();
    } else if (pressed === 't') {
      toggleDrawer();
    } else {
      const medium = config.media.find((candidate) => candidate.key.toLowerCase() === pressed);
      if (medium) {
        selectMedium(medium.id);
        actions.onMedium(medium.id);
      }
    }
  });

  return {
    setStatus(tone, text) {
      status.dataset.tone = tone;
      status.textContent = text;
    },
    setFill(fraction) {
      const percent = Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
      // Drives both the column's height and where the reading rides, so the
      // number always sits at the level it is reporting.
      meter.style.setProperty('--fill', `${percent}%`);
      meterValue.style.setProperty('--fill', `${percent}%`);
      meterValue.textContent = `${percent}%`;
    },
    setSpout(x, y, depth, live) {
      spout.style.transform = `translate(${x}px, ${y}px)`;
      spout.style.setProperty('--ring', `${(14 + depth * 30).toFixed(1)}px`);
      spout.dataset.live = String(live);
    },
    setScene: selectScene,
    setScenario: selectScenario,
    setPouring,

    setTune(key, value) {
      tuneFields.get(key)?.(value);
    },
    setStorm,
    setMedium: selectMedium,
    dismissHint() {
      hint.dataset.hidden = 'true';
    },
  };
}
