import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Healthcare SepsisAI | Research workspace', description: 'Synthetic patient timelines, reproducible risk models and traceable research workflows.' };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
