import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Doctor's Office - Appointment Booking",
  description: "Book and manage your medical appointments",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
