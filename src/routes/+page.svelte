<script lang="ts">
  import { scrapeFood, type ScrapeResult } from '$lib/api';

  let food = 'ginger';
  let maxFoods = 3;
  let maxCompounds = 25;
  let loading = false;
  let error = '';
  let result: ScrapeResult | null = null;

  const examples = ['ginger', 'mango', 'cocoa', 'lemon peel', 'black pepper', 'tea'];

  async function runScrape(value = food) {
    const query = value.trim();
    if (!query) return;
    food = query;
    loading = true;
    error = '';

    try {
      result = await scrapeFood(query, maxFoods, maxCompounds);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Could not scrape the food profile.';
    } finally {
      loading = false;
    }
  }

  function exportJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${result.query.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-flavor-profile.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
</script>

<svelte:head>
  <title>FlavorDB Workbench</title>
  <meta
    name="description"
    content="Scrape FSBI food data into clean structured flavor and molecule output."
  />
</svelte:head>

<main class="shell">
  <section class="hero">
    <div>
      <p class="eyebrow">FSBI / FlavorDB Scraper</p>
      <h1>Food in. Structured flavor chemistry out.</h1>
      <p class="lede">
        Enter a food or aromatic ingredient, scrape FSBI-derived sources, and review normalized foods,
        flavor terms, molecules, and source trace in one clean workspace.
      </p>
    </div>
    <form class="search" on:submit|preventDefault={() => runScrape()}>
      <label>
        <span>Food input</span>
        <input bind:value={food} placeholder="e.g. ginger, mango, cocoa, lemon peel" />
      </label>
      <div class="controls">
        <label>
          <span>Foods</span>
          <input type="number" min="1" max="8" bind:value={maxFoods} />
        </label>
        <label>
          <span>Compounds</span>
          <input type="number" min="1" max="80" bind:value={maxCompounds} />
        </label>
        <button disabled={loading}>{loading ? 'Scraping...' : 'Scrape'}</button>
      </div>
      <div class="examples">
        {#each examples as item}
          <button type="button" on:click={() => runScrape(item)}>{item}</button>
        {/each}
      </div>
    </form>
  </section>

  {#if error}
    <section class="notice error">{error}</section>
  {/if}

  {#if loading}
    <section class="loading">
      <div></div>
      <p>Scraping FSBI pages and normalizing food, molecule, and flavor fields...</p>
    </section>
  {/if}

  {#if result}
    <section class="summary">
      <div>
        <span>Food matches</span>
        <strong>{result.summary.food_matches}</strong>
      </div>
      <div>
        <span>Molecules</span>
        <strong>{result.summary.molecule_matches}</strong>
      </div>
      <div>
        <span>Flavor terms</span>
        <strong>{result.summary.unique_flavor_terms}</strong>
      </div>
      <button on:click={exportJson}>Export JSON</button>
    </section>

    {#if result.warnings.length}
      <section class="notice">
        {#each result.warnings as warning}
          <p>{warning}</p>
        {/each}
      </section>
    {/if}

    <section class="workspace">
      <section class="panel">
        <div class="panelHead">
          <p class="eyebrow">Food records</p>
          <h2>{result.query}</h2>
        </div>
        {#if result.foods.length === 0}
          <p class="muted">No parsed food records returned.</p>
        {:else}
          <div class="foodList">
            {#each result.foods as item}
              <article>
                <h3>{item.alias || result.query}</h3>
                <p>{item.scientific_name || 'Scientific name unavailable'}</p>
                <div class="chips">
                  {#if item.category}<span>{item.category}</span>{/if}
                  <span>{item.molecule_ids.length} linked molecules</span>
                  {#if item.entity_id}<span>FSBI #{item.entity_id}</span>{/if}
                </div>
                <small>{item.source_url}</small>
              </article>
            {/each}
          </div>
        {/if}
      </section>

      <section class="panel">
        <div class="panelHead">
          <p class="eyebrow">Flavor profile</p>
          <h2>Dominant descriptors</h2>
        </div>
        {#if result.flavor_profile.length === 0}
          <p class="muted">No flavor descriptors were extracted for this query.</p>
        {:else}
          <div class="flavorGrid">
            {#each result.flavor_profile.slice(0, 32) as term}
              <div>
                <span>{term.term}</span>
                <strong>{term.count}</strong>
              </div>
            {/each}
          </div>
        {/if}
      </section>
    </section>

    <section class="tablePanel">
      <div class="panelHead">
        <p class="eyebrow">Molecule output</p>
        <h2>Clean structured compounds</h2>
      </div>
      <div class="tableWrap">
        <table>
          <thead>
            <tr>
              <th>PubChem</th>
              <th>Common name</th>
              <th>Flavor profile</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {#each result.molecules as molecule}
              <tr>
                <td>{molecule.pubchem_id ?? 'N/A'}</td>
                <td>{molecule.common_name || 'Unnamed compound'}</td>
                <td>
                  <div class="chips">
                    {#each molecule.flavor_profile as term}
                      <span>{term}</span>
                    {/each}
                  </div>
                </td>
                <td><a href={molecule.source_url} target="_blank" rel="noreferrer">FSBI</a></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <section class="trace">
      <div>
        <p class="eyebrow">Source trace</p>
        <h2>Scrape path</h2>
      </div>
      <ol>
        {#each result.trace as item}
          <li>{item}</li>
        {/each}
      </ol>
    </section>
  {/if}
</main>
