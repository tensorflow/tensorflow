import Link from "next/link";
import Navigation from "./components/Navigation";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Navigation />
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-gray-900 mb-4">
              Welcome to Our Medical Practice
            </h1>
            <p className="text-xl text-gray-600">
              Your health is our priority. Book appointments and manage your healthcare journey with ease.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mb-12">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="text-4xl mb-4">📅</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-3">
                Book an Appointment
              </h2>
              <p className="text-gray-600 mb-6">
                Schedule your visit with our experienced healthcare professionals. Choose your preferred date and time.
              </p>
              <Link
                href="/book"
                className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Book Now
              </Link>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="text-4xl mb-4">📋</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-3">
                View Appointments
              </h2>
              <p className="text-gray-600 mb-6">
                Check your upcoming appointments, reschedule, or book follow-up visits with your doctor.
              </p>
              <Link
                href="/appointments"
                className="inline-block bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
              >
                View Appointments
              </Link>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Our Services</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl mb-3">🩺</div>
                <h3 className="font-semibold text-gray-900 mb-2">General Checkup</h3>
                <p className="text-sm text-gray-600">Comprehensive health examinations</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-3">💉</div>
                <h3 className="font-semibold text-gray-900 mb-2">Vaccinations</h3>
                <p className="text-sm text-gray-600">Stay protected with immunizations</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-3">🔬</div>
                <h3 className="font-semibold text-gray-900 mb-2">Lab Tests</h3>
                <p className="text-sm text-gray-600">Accurate diagnostic testing</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
