# Brand

A separation lane read top to bottom. Each band sits where its affinity for the
sorbent left it: the fraction the phase holds is high and saturated, whatever
had less affinity has run further and faded. Bands carry a soft vertical
falloff because a real chromatographic band has a Gaussian profile, not a hard
edge.

| file | use |
|---|---|
| `logo-light.svg` / `logo-dark.svg` | README header, slides, docs. 440×112. |
| `mark-light.svg` / `mark-dark.svg` | Favicon, avatar, social card. 64×64 square. |

Pair the two themes with `<picture>` so GitHub switches them:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.svg">
  <img alt="Sorbent — compound triage" src="docs/brand/logo-light.svg" width="360">
</picture>
```

## Palette

| role | light | dark |
|---|---|---|
| ink — wordmark, lane, unretained bands | `#0B1220` | `#E6EDF3` |
| held — the retained band | `#A9761B` | `#E0A83C` |
| tagline | `#64748B` | `#8B949E` |

One accent, and it carries the whole idea: gold is the fraction you keep,
everything else is neutral and fading. Do not add a second accent colour.

The wordmark is set in the viewer's own UI sans at weight 300 with wide
tracking, so there is no font to ship and nothing to load. The mark stays
legible at 24 px — the gold band is what survives, so keep it if you ever
simplify further.
