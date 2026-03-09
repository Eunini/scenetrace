import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";

test("workspace, navigation, privacy information and keyboard access are usable", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Find the shot. Keep the evidence." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Explore sample" }),
  ).toBeEnabled();
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to workspace" }),
  ).toBeFocused();
  const report = await new AxeBuilder({ page }).analyze();
  expect(report.violations).toEqual([]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await mkdir("../evidence/screenshots", { recursive: true });
  await page.screenshot({
    path: `../evidence/screenshots/workspace-${info.project.name}.png`,
    fullPage: true,
  });
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "A small model. An inspectable experiment.",
    }),
  ).toBeVisible();
  await expect(
    page.getByText("Release gate passed", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "How it works", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your visitor workspace" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("real sample indexing, model-routed search and an approved playable MP4 export", async ({
  page,
}, info) => {
  test.skip(
    info.project.name === "mobile",
    "The media integration runs once; mobile has separate layout and accessibility coverage.",
  );
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Explore sample" }).click();
  await expect(
    page.getByText("Video indexing · ready", { exact: false }),
  ).toBeVisible({ timeout: 180000 });
  await page
    .getByRole("textbox", { name: "Ask about your footage" })
    .fill("Find footage showing a rabbit");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByRole("button", { name: /Play result 1 at/ }),
  ).toBeVisible({ timeout: 120000 });
  await expect(
    page.getByText(
      "Ranked the closest sampled frames. Play a result to verify the scene.",
    ),
  ).toBeVisible();
  const video = page.getByLabel("Selected source video");
  await expect(video).toBeVisible();
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.readyState), {
      timeout: 30000,
    })
    .toBeGreaterThan(0);
  const duration = await video.evaluate((v: HTMLVideoElement) => v.duration);
  expect(duration).toBeGreaterThan(5);
  await page.getByLabel("Start (seconds)").fill("1");
  await page.getByLabel("End (seconds)").fill("4");
  await page
    .getByRole("button", { name: "Prepare export", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Export 00:01–00:04?" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Download MP4" })).toHaveCount(0);
  await page.getByRole("button", { name: "Approve and export" }).click();
  await expect(page.getByRole("link", { name: "Download MP4" })).toBeVisible({
    timeout: 60000,
  });
  await page.reload();
  await expect(page.getByRole("link", { name: "Download MP4" })).toBeVisible();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "Download MP4" }).click(),
  ]);
  const path = await download.path();
  expect(path).toBeTruthy();
  const output = JSON.parse(
    execFileSync(
      "ffprobe",
      ["-v", "error", "-show_format", "-of", "json", path!],
      { encoding: "utf8" },
    ),
  );
  expect(Number(output.format.duration)).toBeGreaterThanOrEqual(2.9);
  expect(Number(output.format.duration)).toBeLessThanOrEqual(3.2);
  await page.screenshot({
    path: "../evidence/screenshots/evidence-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Search was useful" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Feedback saved" }),
  ).toBeVisible();
  page.on("dialog", (dialog) => dialog.accept());
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await expect(
    page.getByText("Workspace cleared.", { exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});
