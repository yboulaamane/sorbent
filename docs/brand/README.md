# Brand

A wordmark, and one line beneath it. The gold segment is the fraction the phase
holds; the faint remainder is what ran past. That is the whole idea, and it is
the only thing the accent colour is ever used for.

| file | use |
|---|---|
| `logo-light.svg` / `logo-dark.svg` | README header, slides, docs. 212×80. |
| `mark-light.svg` / `mark-dark.svg` | Favicon, avatar, social card. 64×64 square. |

Pair the two themes with `<picture>` so GitHub switches them:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.svg">
  <img alt="Sorbent — compound triage" src="docs/brand/logo-light.svg" width="260">
</picture>
```

## Palette

| role | light | dark |
|---|---|---|
| ink — wordmark, unretained band | `#0B1220` | `#E6EDF3` |
| held — the retained fraction | `#A9761B` | `#E0A83C` |
| tagline | `#64748B` | `#8B949E` |

One accent, carrying the whole idea. Do not add a second.

## Two constraints worth keeping

**No `<tspan>`, and no `text-anchor="middle"` on text.** Some renderers treat
each tspan as its own text chunk, which drops the letter-spacing either side of
it; combined with `text-anchor="middle"` they re-anchor every chunk at the same
point and the letters stack on top of each other. An earlier draft coloured the
`O` gold that way and rendered as `RSOBENT`. The accent is geometry instead, so
the mark looks the same in every renderer.

**No font is shipped.** The wordmark is set in the viewer's own UI sans at
weight 400 with wide tracking, so nothing loads and nothing can fail to load.
It resolves to SF Pro, Segoe UI or similar; on a bare Linux box it falls back
to DejaVu Sans, which is the worst case and still holds.

The square mark stays legible at 24 px. The gold band is what survives, so keep
it if you ever simplify further.
